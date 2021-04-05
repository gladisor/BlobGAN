from models import DCGAN
from utils import BlobData
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# path = Path().absolute()
# test = BlobData(path / 'data' / 'test')
# x, y = test[0]
#
transform = transforms.Compose([
    transforms.ToPILImage(),
    ])

dcgan = DCGAN(
    z_channels=100, h_channels=32,
    img_channels=1, lr=0.0002)

z = dcgan.get_noise(10)
x = dcgan.G(z)

print(x.shape)

y = dcgan.D(x)
print(y.shape)

# imgs = []
# for i in range(x.shape[0]):
#     imgs.append(transform(x[i]))
#
# print(imgs)
