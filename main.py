from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import ImageData
from models import CNN, Classifier
import pytorch_lightning as pl

if __name__ == '__main__':
    path = Path().resolve()

    train = ImageData(torch.load(path / 'MNIST/processed/training.pt'))
    test = ImageData(torch.load(path / 'MNIST/processed/test.pt'))

    # idx = torch.randint(high=len(train), size=(1,)).item()
    # img, label = train[idx]
    #
    # print(label)
    # plt.imshow(img[0])
    # plt.show()

    model = CNN(train.n_class)
    classifier = Classifier(model)

    batch_size = 32
    train = DataLoader(train, batch_size=batch_size, num_workers=1, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, num_workers=1)

    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(classifier, train, test)
