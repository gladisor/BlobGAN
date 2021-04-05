from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils import BlobData
from models import CNN, Classifier, DCGAN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# from torchvision.transforms import ToPILImage

if __name__ == '__main__':
    path = Path().resolve()
    train_path = path / 'data' / 'train'
    # test_path = path / 'data' / 'test'

    train = BlobData(train_path)
    # test = BlobData(test_path)

    # model = CNN(train.n_class)
    # classifier = Classifier(
    #     model=model,
    #     lr=1e-3)

    batch_size = 128
    train = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
    # test = DataLoader(test, batch_size=batch_size, num_workers=1)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=3,
        logger=WandbLogger(project='BlobGAN'),
        log_every_n_steps=1
        )

    dcgan = DCGAN(
        z_channels=100, h_channels=32,
        img_channels=1, lr=0.0002)

    # trainer.fit(classifier, train, test)
    trainer.fit(dcgan, train)
