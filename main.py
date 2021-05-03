from pathlib import Path
from torch.utils.data import DataLoader

from utils import BlobData
from models import DCGAN

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

## https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    def __init__(self, save_every: int):
        self.save_every = save_every

    def on_batch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_every == 0:
            filename = f'{epoch=}_{global_step=}.ckpt'
            ckpt_path = Path(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

if __name__ == '__main__':
    mode = '1' ## ['1', 'RGB']

    path = Path().resolve()
    train_path = path / 'data' / mode / 'train'

    mode_channels = {'1':1, 'RGB':3}

    z_channels = 100
    h_channels = 64
    img_channels = mode_channels[mode]
    lr = 0.0002
    batch_size = 256

    ## Defining model
    dcgan = DCGAN(
        z_channels=z_channels, h_channels=h_channels,
        img_channels=img_channels, lr=lr,
        save_every=200)

    ## Creating dataset object
    train = BlobData(train_path, img_channels)
    print(f'Number of training examples is {len(train)}')

    train = DataLoader(
        train, batch_size=batch_size,
        num_workers=4, shuffle=True)

    ## Making logger for tracking stats on wandb.com
    logger = WandbLogger(
        project='BlobGAN',
        name=f'img={img_channels}-h={h_channels}-z={z_channels}-lr={lr}-batch={batch_size}',
        log_model=False)

    ## Creating trainer object
    callback = CheckpointEveryNSteps(dcgan.save_every)
    trainer = Trainer(
        gpus=1,
        max_epochs=2,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[callback])

    ## Fitting model
    trainer.fit(dcgan, train)
