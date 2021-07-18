from torchvision.transforms import transforms
from module.gaussian_blur import GaussianBlur

def train_transform(size=224):
    color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          color_jitter,
                                          transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 2.0)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
    return data_transforms

def val_test_transform(size=224):
    data_transforms = transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
    return data_transforms
