from module.data_aug import train_transform, val_test_transform
import torch
from torch.utils.data import dataset
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from SimCLR import SimCLR
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform

train_transform = train_transform(size=224)
val_test_transform = val_test_transform(size=224)

train_data = CIFAR10(download=True,root="./cifar10",transform=SimCLRTrainDataTransform())
test_val_data = CIFAR10(root="./cifar10",train = False,transform=SimCLREvalDataTransform())
train_len = len(train_data)
val_len = test_len = int(len(test_val_data)/2)
test_data, val_data = torch.utils.data.random_split(test_val_data, [test_len, val_len])
num_class = len(np.unique(train_data.targets))
train_loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True, drop_last=True, pin_memory=False, num_workers=16)
test_loader = DataLoader(dataset = test_data, batch_size = 16)
valid_loader = DataLoader(dataset = val_data, batch_size= 16)

checkpoint_callback = ModelCheckpoint(
    monitor='avg_train_loss',
    dirpath='./models/',
    filename='simclr-{epoch:02d}-{avg_train_loss:.2f}',
    mode='min',
    save_last=True,
    every_n_train_steps=10
)

simclr = SimCLR(gpus=2)
trainer = pl.Trainer(gpus=2, accelerator='ddp', callbacks=[checkpoint_callback])
trainer.fit(simclr, train_loader)

