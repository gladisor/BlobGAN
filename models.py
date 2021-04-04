import torch
import torch.nn as nn
import pytorch_lightning as pl

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
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.model(x)
        return x

    def accuracy(self, y_hat, y):
        prediction = torch.argmax(y_hat, dim=-1)
        correct = (prediction == y)
        acc = correct.sum() / correct.shape[0]
        return acc

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log('validation_loss', loss)
        self.log('validation_acc', acc)
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
