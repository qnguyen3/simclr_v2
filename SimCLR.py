import math
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn, Tensor
from torch.nn import functional as F
import module.resnet as resnet
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from ContrastiveLoss import ContrastiveLoss

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(LightningModule):

    def __init__(
        self,
        num_samples: int = 16,
        batch_size: int = 32,
        gpus: int = 1,
        num_nodes: int = 1,
        arch: str = 'resnet18', #
        hidden_mlp: int = 2048, #
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        mode: str = None,
        maxpool1: bool = True,
        optimizer: str = 'lars',
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.mode = mode
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_model()
        if arch = 'resnet50':
            self.features = nn.Linear(self.hidden_mlp, self.hidden_mlp) #First Projection Head
            self.batch_norm1d = nn.BatchNorm1d(self.hidden_mlp)
            self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)
        elif arch = 'resnet18':
            self.features = nn.Linear(512, 512) #First Projection Head
            self.batch_norm1d = nn.BatchNorm1d(512)
            self.projection = Projection(input_dim=512, hidden_dim=512, output_dim=self.feat_dim)

        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet.resnet18(mode=self.mode)
        elif self.arch == 'resnet50':
            backbone = resnet.resnet50(mode=self.mode)

        return backbone
    
    def forward(self, x):
        x = self.encoder(x)
        return self.features(x)

    def shared_step(self, batch):
        x, y = batch
        batch_size = y.shape[0]//2
        features = self(x)
        features = self.features(features)
        features = F.relu(self.batch_norm1d(features))
        features = self.projection(features)
        # f1, f2 = torch.split(features, (batch_size,batch_size), dim=0)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        features = features.unsqueeze(0)
        loss = self.loss_function(features)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def loss_function(self, x):
        loss_fn = ContrastiveLoss()
        loss = loss_fn(x)
        return loss
    
    
    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate*10,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'adamw':
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    

    
    

