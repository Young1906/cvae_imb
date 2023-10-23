import lightning as L

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from modules.cvae.net import build_mlp
from modules.cvae.net import Dense


class Encoder(nn.Module):
    """
    Description:
        MLP encoder 

    Args:
        - input_dim: dimension of input vector
        - seq: [(h0, a0), ...] a tuple describe and mlp;
            in which hi, ai is the number of hidden unit
            and activation function at layer i, repectively
        - z_dim: latent variable dimension

    Return:
        - mu, sigma
    """
    def __init__(self, input_dim: int, seq: list[list], z_dim: int):
        super().__init__()

        self.mlp = build_mlp(input_dim, seq)
        d_mlp, _ = seq[-1]

        # Dense layer return mu & logvar
        self.mu = Dense(d_mlp, z_dim, 'tanh')
        self.logvar= Dense(d_mlp, z_dim, 'tanh')


    def forward(self, x):
        z = self.mlp(x)
        mu, logvar = self.mu(z), self.logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    """
    Description:
        MLP decoder 

    Args:
        - input_dim: dimension of latent 
        - seq: [(h0, a0), ...] a tuple describe and mlp;
            in which hi, ai is the number of hidden unit
            and activation function at layer i, repectively
        - output_dim: input vector size

    Return:
        - mu, sigma
    """
    def __init__(
            self,
            input_dim: int,
            seq: list[list],
            output_dim: int,):
        super().__init__()

        self.mlp = build_mlp(input_dim, seq)
        d_mlp, _ = seq[-1]
        self.out = Dense(d_mlp, output_dim, 'tanh')


    def forward(self, z):
        return self.out(self.mlp(z))


class CVAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            encoder: str,
            decoder: str,
            z_dim: int,
            n_class: int):
        super().__init__()
        self.encoder = Encoder(input_dim + n_class, encoder, z_dim)
        self.decoder = Decoder(z_dim + n_class, decoder, input_dim)
        self.n_class = n_class


    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.n_class)

        mu, logvar = self.encoder(torch.concat([x, y], -1))
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(torch.concat([z, y], -1))

        return x_res, mu, logvar 


class LightCVAE(L.LightningModule):
    def __init__(self,
                 input_dim: int,
                 encoder: list[list],
                 decoder: list[list],
                 z_dim: int,
                 n_class: int):
        super().__init__()
        self.save_hyperparameters()
        self.cvae = CVAE(input_dim, encoder, decoder, z_dim, n_class)

    def training_step(self, batch, batch_idx):
        # unpacking
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)

        loss = self.elbo(x, x_res, mu, logvar)
        self.log("train-loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)

        loss = self.elbo(x, x_res, mu, logvar)
        self.log("valid-loss", loss, prog_bar=True)
        return loss

    @staticmethod
    def elbo(x, x_res, mu, logvar):
        # reconstruction loss
        lR = ((x - x_res)**2).mean()
        kL = -.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp())
        return 0.5 * lR + 0.5 * kL

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
