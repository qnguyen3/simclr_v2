from module.data_aug import train_transform, val_test_transform
import torch
from torch.utils.data import dataset
import numpy as np
from torchvision.datasets import CIFAR10, ImageNet
from torch.utils.data.dataloader import DataLoader
from SimCLR import SimCLR
import pytorch_lightning as pl
CUDA_LAUNCH_BLOCKING=1
from pytorch_lightning.callbacks import ModelCheckpoint
# from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
from module.simclr_transform import get_simclr_data_transforms_test, get_simclr_data_transforms_train
from module.multi_view_data_injector import MultiViewDataInjector
train_transform = get_simclr_data_transforms_train('cifar10')
val_test_transform = get_simclr_data_transforms_test('cifar10')

train_data = CIFAR10(download=True,root="./cifar10",transform=MultiViewDataInjector([train_transform,train_transform,val_test_transform]))
train_len = len(train_data)
num_class = len(np.unique(train_data.targets))
train_loader = DataLoader(dataset = train_data, batch_size = 16)
# test_loader = DataLoader(dataset = test_data, batch_size = 16)
# valid_loader = DataLoader(dataset = val_data, batch_size= 16)

checkpoint_callback = ModelCheckpoint(
    dirpath='./models/',
    monitor = 'avg_train_loss',
    filename='simclr-{epoch:02d}-{avg_train_loss:.2f}',
    mode='min',
    save_last=True,
)

simclr = SimCLR(arch='resnet18',mode='cifar10',gpus=1)
trainer = pl.Trainer(callbacks=[checkpoint_callback],gpus=1)
trainer.fit(simclr, train_loader)

