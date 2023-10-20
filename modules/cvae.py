import lightning as L
import torch

from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class Dense(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, act: str):
        super().__init__()
        
        if act == "sigmoid":
            act = nn.Sigmoid() 

        elif act == "relu":
            act = nn.ReLU()

        elif act == "leaky_relu":
            act = nn.LeakyReLU()
        
        elif act == "tanh":
            act = nn.Tanh()
        
        else:
            NotImplementedError(act)

        self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act)

    def forward(self, x):
        return self.seq(x)



class Encoder(nn.Module):
    """
    MLP encoder 
    """
    def __init__(self, input_dim: int, dims: list[int], z_dim: int):
        super().__init__()

        seqs = []
        seqs.append(Dense(input_dim, dims[0], 'relu'))
        
        for i in range(0, len(dims)-1):
            seqs.append(Dense(dims[i], dims[i+1], 'relu'))

        #seqs.append(Dense(dims[-1], z_dim, 'relu'))
        self.seqs = nn.Sequential(*seqs)

        self.mu = Dense(dims[-1], z_dim, 'tanh')
        self.logvar= Dense(dims[-1], z_dim, 'tanh')


    def forward(self, x):
        z = self.seqs(x)
        mu, logvar = self.mu(z), self.logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    """
    MLP encoder 
    """
    def __init__(self, z_dim: int, dims: list[int], output_dims: int):
        super().__init__()

        seqs = []
        seqs.append(Dense(z_dim, dims[0], 'relu'))
        
        for i in range(0, len(dims)-1):
            seqs.append(Dense(dims[i], dims[i+1], 'relu'))

        seqs.append(Dense(dims[-1], output_dims, 'relu'))
        self.seqs = nn.Sequential(*seqs)

    def forward(self, x):
        return self.seqs(x)


class CVAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            dims: list[int],
            z_dim: int,
            n_class: int):
        super().__init__()
        self.encoder = Encoder(input_dim + n_class, dims, z_dim)
        self.decoder = Decoder(z_dim, dims[::-1], input_dim)
        self.n_class = n_class


    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.n_class)

        mu, logvar = self.encoder(torch.concat([x, y], -1))
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(z)

        return x_res, mu, logvar 


class LightCVAE(L.LightningModule):
    def __init__(self,
                 input_dim: int,
                 dims: list[int],
                 z_dim: int,
                 n_class: int):
        super().__init__()
        self.cvae = CVAE(input_dim, dims, z_dim, n_class)

    def training_step(self, batch, batch_idx):
        return 

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers():
        return torch.optim.Adam(self.parameters(), lr=1e-2)



if __name__ == "__main__":
    vae = CVAE(128, [64, 32, 16], 16, 10)
    x = torch.randn((256, 128))
    y = torch.randint(low=0, high=10, size=(256,))
    x_res, mu, logvar = vae(x, y)

