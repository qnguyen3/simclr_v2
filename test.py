import torch
import module.resnet as resnet
import torchvision.models as models
import torchvision
import torch.nn as nn
from ContrastiveLoss import ContrastiveLoss
from SimCLR import Projection, SimCLR
from torchvision.transforms import transforms

resnet50 = resnet.resnet50()

torchrn50 = models.resnet50(pretrained=False)


image = torchvision.io.read_image(r'Y:\github\SimCLR\cat_224x224.jpg')
image = image.unsqueeze(0)
image2 = torchvision.io.read_image(r'Y:\github\SimCLR\unnamed.jpg')
image2 = image2.unsqueeze(0)
images = torch.cat((image, image2))
print(images.shape)
images = images/255
output = resnet50(images)
# print(output.shape)
# loss = ContrastiveLoss()
# f1, f2 = torch.split(output, [1, 1], dim=0)
# features = output.unsqueeze(0)
# # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
# print(features.shape)
# loss_item = loss(features)
# print(loss_item)
# loss_item.backward()
# print(loss_item)
# print(resnet50)

# proj = Projection(2048, 2048, 128)
# x = proj(output)
# x = x.unsqueeze(0)
# print(x.shape)
# loss = ContrastiveLoss()
# loss_item = loss(x)
# print(loss_item)

simclr = SimCLR(1,64,None)

print(simclr)

output  = simclr(images)
print(output.shape)