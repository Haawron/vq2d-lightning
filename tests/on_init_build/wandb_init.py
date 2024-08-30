# This script needs these libraries to be installed:
#   torch, torchvision, pytorch_lightning

import wandb

import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, lr=1e-3, inp_size=28, optimizer = 'Adam'):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size))
        self.lr = lr

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        # log metrics to wandb
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

# setup data
batch_size = 32
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, shuffle=True)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project='my-awesome-project')

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# pass wandb_logger to the Trainer
trainer = pl.Trainer(limit_train_batches=750, max_epochs=5, logger=wandb_logger)

# train the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
