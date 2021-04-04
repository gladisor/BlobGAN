from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import BlobData
from models import CNN, Classifier
import pytorch_lightning as pl

if __name__ == '__main__':
    path = Path().resolve()
    train_path = path / 'data' / 'train'
    test_path = path / 'data' / 'test'

    train = BlobData(train_path)
    test = BlobData(test_path)

    # print(train[0])

    print(train.n_class)
    model = CNN(train.n_class)
    classifier = Classifier(model)

    batch_size = 32
    train = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, num_workers=1)

    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(classifier, train, test)
