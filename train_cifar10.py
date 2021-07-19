from module.data_aug import train_transform, val_test_transform
import torch
from torch.utils.data import dataset
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from SimCLR import SimCLR
import pytorch_lightning as pl
from module.gaussian_blur import GaussianBlur
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform

train_transform = train_transform(size=224)
val_test_transform = val_test_transform(size=224)

train_data = CIFAR10(download=True,root="./cifar10",transform=SimCLRTrainDataTransform())
test_val_data = CIFAR10(root="./cifar10",train = False,transform=val_test_transform)
train_len = len(train_data)
val_len = test_len = int(len(test_val_data)/2)
test_data, val_data = torch.utils.data.random_split(test_val_data, [test_len, val_len])
num_class = len(np.unique(train_data.targets))
train_loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True, drop_last=True, pin_memory=True, num_workers=24)
test_loader = DataLoader(dataset = test_data, batch_size = 16, shuffle = True, drop_last=True, pin_memory=False, num_workers=24)
valid_loader = DataLoader(dataset = val_data, batch_size= 16)

checkpoint_callback = ModelCheckpoint(
    monitor='avg_train_loss',
    dirpath='./models/',
    filename='simclr-{epoch:02d}-{avg_train_loss:.2f}',
    mode='min',
)

simclr = SimCLR(gpus=2, num_nodes=1)
trainer = pl.Trainer(gpus=2, accelerator='ddp', callbacks=[checkpoint_callback], num_nodes=1)
trainer.fit(simclr, test_loader)

