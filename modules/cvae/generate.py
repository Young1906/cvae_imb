import unittest

import torch
from torch import nn
from torch.nn import functional as F
from modules.cvae.cvae import LightCVAE
from modules.base import Layer

def load_decoder(pth: str,
                 input_dim: int,
                 encoder: list[Layer],
                 decoder: list[Layer],
                 z_dim: int,
                 n_class: int) -> nn.Module:
    """
    """
    model = LightCVAE.load_from_checkpoint(
            pth,
            input_dim=input_dim,
            encoder=encoder,
            decoder=decoder,
            z_dim=z_dim,
            n_class=n_class)
    model.eval()
    return model.cvae.decoder


def load_encoder(pth: str,
                 input_dim: int,
                 encoder: list[Layer],
                 decoder: list[Layer],
                 z_dim: int,
                 n_class: int) -> nn.Module:
    """
    """
    model = LightCVAE.load_from_checkpoint(
            pth,
            input_dim=input_dim,
            encoder=encoder,
            decoder=decoder,
            z_dim=z_dim,
            n_class=n_class)
    model.eval()
    return model.cvae.encoder


def generate(decoder, n_samples, z_dim, y, n_class):
    y = F.one_hot(y, num_classes=n_class)
    z = torch.randn(size=[n_samples, z_dim])
    samples = decoder(torch.concat([z, y], -1))
    return samples.detach().cpu().numpy()


class TestLoadDecoder(unittest.TestCase):
    def test_generate(self):
        decoder = load_decoder(
                "logs/dev/version_0/checkpoints/epoch=99-step=2000.ckpt",
                input_dim=32,
                encoder="mlp_16_8_16",
                decoder="mlp_16_8_16",
                z_dim=8,
                n_class=2)

        n_samples = 16 
        z_dim = 8
        y = torch.randint(2, size=[16,])

        samples = generate(decoder, n_samples, z_dim, y, 2)
        self.assertEqual(samples.size(), (16, 32))

if __name__ == "__main__":
    unittest.main()
