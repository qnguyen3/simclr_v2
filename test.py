# from torchvision.datasets import CIFAR10
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import Subset
# import torch
# import numpy as np
# from pl_bolts.models.self_supervised import SSLFineTuner
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from SimCLR import SimCLR
# from pytorch_lightning import callbacks
# from module.data_aug import train_transform, val_test_transform
# backbone = SimCLR.load_from_checkpoint('./models/simclr-epoch=00-avg_train_loss=0.00-v1.ckpt', strict=False)

# train_transform = train_transform(size=224)
# val_test_transform = val_test_transform(size=224)

# train_data = CIFAR10(download=True,root="./cifar10",transform=train_transform)
# test_val_data = CIFAR10(root="./cifar10",train = False,transform=val_test_transform)
# val_len = test_len = int(len(test_val_data)/2)
# test_data, val_data = torch.utils.data.random_split(test_val_data, [test_len, val_len])
# num_class = len(np.unique(train_data.targets))
# train_loader = DataLoader(dataset = train_data, batch_size = 256, shuffle = True, num_workers = 12, drop_last=True, pin_memory=True)
# test_loader = DataLoader(dataset = test_data, batch_size= 32, pin_memory=True)
# valid_loader = DataLoader(dataset = val_data, batch_size= 32, drop_last=True,pin_memory=True)

# finetuner = SSLFineTuner(backbone, in_features=backbone.hidden_mlp, num_classes=num_class, hidden_dim=1024)
# checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',
#     dirpath='./models/',
#     filename='simclr_finetune-{epoch:02d}-{val_loss:.2f}',
#     mode='min',
# )
# trainer = pl.Trainer(gpus=1,callbacks=[checkpoint_callback])
# # trainer.fit(finetuner, train_loader, valid_loader)
# trainer.test(model=finetuner, ckpt_path='./models/simclr_finetune-epoch=130-val_loss=0.85.ckpt', test_dataloaders=test_loader)



# # import torch
# # import module.resnet as resnet
# # import torchvision.models as models
import torchvision
# # import torch.nn as nn
# # from ContrastiveLoss import ContrastiveLoss
# # from SimCLR import Projection, SimCLR
# # from torchvision.transforms import transforms

# # resnet50 = resnet.resnet50()

# # torchrn50 = models.resnet50(pretrained=False)


# # image = torchvision.io.read_image(r'Y:\github\SimCLR\cat_224x224.jpg')
# # image = image.unsqueeze(0)
# # image2 = torchvision.io.read_image(r'Y:\github\SimCLR\unnamed.jpg')
# # image2 = image2.unsqueeze(0)
# # images = torch.cat((image, image2))
# # print(images.shape)
# # images = images/255
# # output = resnet50(images)
# # # print(output.shape)
# # # loss = ContrastiveLoss()
# # # f1, f2 = torch.split(output, [1, 1], dim=0)
# # # features = output.unsqueeze(0)
# # # # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
# # # print(features.shape)
# # # loss_item = loss(features)
# # # print(loss_item)
# # # loss_item.backward()
# # # print(loss_item)
# # # print(resnet50)

# # # proj = Projection(2048, 2048, 128)
# # # x = proj(output)
# # # x = x.unsqueeze(0)
# # # print(x.shape)
# # # loss = ContrastiveLoss()
# # # loss_item = loss(x)
# # # print(loss_item)

# # simclr = SimCLR(1,64,None)

# # print(simclr)

# # output  = simclr(images)
# # print(output.shape)
from module.resnet import ResNetPreTrained
import torch
model = ResNetPreTrained()
print(model)
image = torchvision.io.read_image(r'Y:\github\SimCLR_v2_pub\simclr_v2\unnamed.jpg')
image = image.unsqueeze(0)
output = model(image.float())
features = torch.nn.Linear(512, 2048)
out = features(output.flatten())
print(output.flatten().shape)

