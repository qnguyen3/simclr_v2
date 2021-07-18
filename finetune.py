from pytorch_lightning import callbacks
from module.data_aug import train_transform, val_test_transform
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch
import numpy as np
from pl_bolts.models.self_supervised import SSLFineTuner
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from SimCLR import SimCLR

backbone = SimCLR.load_from_checkpoint('./saved_models/simclr-epoch=12-avg_train_loss=4.86.ckpt', strict=False)

train_transform = train_transform(size=224)
val_test_transform = val_test_transform(size=224)

train_data = CIFAR10(download=True,root="./cifar10",transform=train_transform)
test_val_data = CIFAR10(root="./cifar10",train = False,transform=val_test_transform)
val_len = test_len = int(len(test_val_data)/2)
test_data, val_data = torch.utils.data.random_split(test_val_data, [test_len, val_len])
num_class = len(np.unique(train_data.targets))
train_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, drop_last=True)
test_loader = DataLoader(dataset = test_data, batch_size=32)
valid_loader = DataLoader(dataset = val_data, batch_size= 8, drop_last=True)

finetuner = SSLFineTuner(backbone, in_features=backbone.hidden_mlp, num_classes=num_class, hidden_dim=1024)
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./models/',
    filename='simclr_finetune-{epoch:02d}-{val_loss:.2f}',
    mode='min',
)
trainer = pl.Trainer(gpus=1,callbacks=[checkpoint_callback])
trainer.fit(finetuner, train_loader, valid_loader)





