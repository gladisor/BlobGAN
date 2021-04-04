from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from PIL import Image

path = Path().absolute()
data = ImageFolder(path, transform=transforms.ToTensor())
print(data[0:1])
