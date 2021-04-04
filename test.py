from utils import BlobData
from models import CNN
from pathlib import Path

path = Path().absolute()

train = BlobData(path / 'data' / 'train')
x, y = train[10]

x = x.unsqueeze(0)

model = CNN(train.n_class)
print(model(x))
