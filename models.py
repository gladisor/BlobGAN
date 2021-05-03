import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torchvision.transforms import ToPILImage
import wandb

## https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Generator(nn.Module):
    """
    Generator convolutional network which takes in a latent vector
    of shape (batch, latent_dim, 1, 1) and produces an image of size (128, 128).
    """
    def __init__(self, z_channels: int, h_channels: int, out_channels: int):
        super().__init__()
        self.z_channels = z_channels
        self.h_channels = h_channels
        self.out_channels = out_channels

        ## Static archetecture parameters
        # self.h_dims = [16, 8, 4, 2, 1]
        self.h_dims = [8, 4, 2, 1]
        self.h_dims = [x * h_channels for x in self.h_dims]
        ## Setting conv parameters
        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.layers = self.create_layers()

    def create_layers(self):
        ## Input layer
        layers = [
            nn.ConvTranspose2d(
                self.z_channels, self.h_dims[0],
                self.kernel, bias=False),
            nn.BatchNorm2d(self.h_dims[0]),
            nn.ReLU()]

        ## Hidden layers
        for i in range(len(self.h_dims) - 1):
            layers += [
                nn.ConvTranspose2d(
                    self.h_dims[i], self.h_dims[i+1],
                    self.kernel, self.stride,
                    self.padding, bias=False),
                nn.BatchNorm2d(self.h_dims[i+1]),
                nn.ReLU()]

        ## Output layer
        layers += [
            nn.ConvTranspose2d(
                self.h_dims[-1], self.out_channels,
                self.kernel, self.stride,
                self.padding, bias=False),
            nn.Tanh()]

        ## Unpacking layers into sequential block
        return nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)

## https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self, in_channels: int, h_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels

        ## Static archetecture parameters
        # self.h_dims = [1, 2, 4, 8, 16]
        self.h_dims = [1, 2, 4, 8]
        self.h_dims = [x * h_channels for x in self.h_dims]
        ## Setting conv parameters
        self.kernel = 4
        self.stride = 2
        self.padding = 1
        ## Slope for leaky ReLU
        self.slope = 0.2

        self.layers = self.create_layers()

    def create_layers(self):
        ## Input layer
        layers = [
            nn.Conv2d(
                self.in_channels, self.h_dims[0],
                self.kernel, self.stride,
                self.padding, bias=False),
            nn.LeakyReLU(self.slope)]

        ## Hidden layers
        for i in range(len(self.h_dims) - 1):
            layers += [
                nn.Conv2d(
                    self.h_dims[i], self.h_dims[i+1],
                    self.kernel, self.stride,
                    self.padding, bias=False),
                nn.BatchNorm2d(self.h_dims[i+1]),
                nn.LeakyReLU(self.slope)]

        ## Output layer
        layers += [
            nn.Conv2d(self.h_dims[-1], 1, self.kernel, bias=False)
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        probs = self.layers(x).squeeze()
        return probs

    def get_features(self, x):
        return self.layers[:-3](x)

## https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
class DCGAN(pl.LightningModule):
    def __init__(self,
            z_channels: int, h_channels: int,
            img_channels: int, lr: float,
            save_every: int = 100):
        super().__init__()
        self.save_hyperparameters()
        ## Hyperparams
        self.z_channels = z_channels
        self.h_channels = h_channels
        self.img_channels = img_channels
        self.lr = lr
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.save_every = save_every

        ## Generator
        self.G = Generator(z_channels, h_channels, img_channels)
        self.G.apply(self.weights_init)

        ## Discriminator
        self.D = Discriminator(img_channels, h_channels)
        self.D.apply(self.weights_init)

        ## Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        ## Turning off automatic optimization
        self.automatic_optimization = False

        ## PIL image tranform
        self.transform = ToPILImage()

        ## Latent vectors for testing
        self.latents = self.get_noise(9)

    def forward(self, x):
        return self.G(x)

    ## https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.z_channels, 1, 1, device=self.device)

    def get_images(self):
        imgs = []
        for i in range(self.latents.shape[0]):
            z = self.latents[i].unsqueeze(0)
            x = self.G(z.to(self.device))
            imgs.append(self.transform(x.squeeze(0)))
        return imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        ## Gathering optimizers
        g_opt, d_opt = self.optimizers()
        ## Sampling batch
        real_x, _ = batch
        ## Creating labels
        batch_size = real_x.shape[0]
        real_label = torch.ones((batch_size,), device=self.device)
        fake_label = torch.zeros((batch_size,), device=self.device)

        z = self.get_noise(batch_size)
        fake_x = self.G(z)

        ## Training discriminator
        d_opt.zero_grad()

        ## Calculating loss on real images
        real_x_preds = self.D(real_x)
        real_loss = self.criterion(real_x_preds, real_label)

        ## Calculating loss on fake images
        fake_x_preds = self.D(fake_x.detach())
        fake_loss = self.criterion(fake_x_preds, fake_label)

        ## Summing loss and stepping optimizer
        d_loss = (real_loss + fake_loss)
        self.manual_backward(d_loss)
        d_opt.step()

        ## Calculating accuracy of discriminator
        real_acc = accuracy(self.sigmoid(real_x_preds), real_label.int())
        fake_acc = accuracy(self.sigmoid(fake_x_preds), fake_label.int())

        ## Training generator
        g_opt.zero_grad()

        ## Calculating generator loss
        fake_x_preds = self.D(fake_x)
        g_loss = self.criterion(fake_x_preds, real_label)
        # g_loss = -fake_x_preds.mean() ## Loss function which supposedly prevents sigmoid saturation

        ## Stepping optimizer
        self.manual_backward(g_loss)
        g_opt.step()

        ## Logging stats
        self.log_dict(
            {'d_loss': d_loss, 'g_loss': g_loss, 'real_acc': real_acc, 'fake_acc': fake_acc},
            prog_bar=True)

        ## Logging images
        if batch_idx % self.save_every == 0:
            imgs = self.get_images()
            imgs = [wandb.Image(img) for img in imgs]
            self.logger.experiment.log({'generated_images': imgs})

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return g_opt, d_opt

class CNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.img_size = 128
        self.in_c = 1
        self.hidden_c = 16
        self.kernel_size = 2

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_c, self.hidden_c, self.kernel_size),
            nn.MaxPool2d(self.kernel_size),
            nn.ReLU(),
            *self.block(4),
            nn.Flatten())

        self.fc = nn.Linear(144, n_class)

    def block(self, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(self.hidden_c, self.hidden_c, self.kernel_size),
                nn.MaxPool2d(self.kernel_size),
                nn.ReLU()]
        return layers

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.conv(x)
        x = self.fc(x)
        return x

class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits.detach(), dim=-1)
        acc = accuracy(probs, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits.detach(), dim=-1)
        acc = accuracy(probs, y)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
