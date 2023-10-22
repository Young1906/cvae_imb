import lightning as L

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from modules.net import build_mlp
from modules.net import Dense


class Encoder(nn.Module):
    """
    MLP encoder 
    """
    def __init__(self, input_dim: int, mlp_name: str, z_dim: int):
        super().__init__()

        if mlp_name == "mlp_16_8_16":
            from modules.net import mlp_16_8_16
            self.mlp = mlp_16_8_16(input_dim)

        else:
            raise NotImplementedError(mlp_name)

        # Dense layer return mu & logvar
        self.mu = Dense(16, z_dim, 'tanh')
        self.logvar= Dense(16, z_dim, 'tanh')


    def forward(self, x):
        z = self.mlp(x)
        mu, logvar = self.mu(z), self.logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    """
    MLP decoder 
    """
    def __init__(self, z_dim: int, mlp_name, output_dim: int, n_class :int):
        super().__init__()

        self.n_class=n_class
        
        if mlp_name == "mlp_16_8_16":
            from modules.net import mlp_16_8_16
            self.mlp = mlp_16_8_16(z_dim + n_class)
        
        else:
            raise NotImplementedError(mlp_name)

        self.out = Dense(16, output_dim, 'tanh')


    def forward(self, z, y):
        return self.out(self.mlp(torch.concat([z, y], -1)))


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
        self.decoder = Decoder(z_dim, decoder, input_dim, n_class)
        self.n_class = n_class


    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.n_class)

        mu, logvar = self.encoder(torch.concat([x, y], -1))
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(z, y)

        return x_res, mu, logvar 


class LightCVAE(L.LightningModule):
    def __init__(self,
                 input_dim: int,
                 encoder: str,
                 decoder: str,
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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)

        loss = self.elbo(x, x_res, mu, logvar)
        self.log("valid-loss", loss, prog_bar=True)

    @staticmethod
    def elbo(x, x_res, mu, logvar):
        # reconstruction loss
        lR = ((x - x_res)**2).mean()
        kL = -.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp())
        return 0.5 * lR + 0.5 * kL

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
