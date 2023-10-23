import torch
import unittest 

from torch import nn
from torch.nn import functional as F


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


def build_mlp(input_dim, seq) -> nn.Module:
    """
    seq: [(h, act), ...]
    """
    net = []
    h0, a0 = seq[0]
    net.append(Dense(input_dim, h0, a0))

    for j in range(1, len(seq)):
        hi, _ = seq[j - 1]
        hj, aj = seq[j]
        net.append(Dense(hi, hj, aj))

    return nn.Sequential(*net)



class TestMLP(unittest.TestCase):
    def test_build_mlp(self):
        net = build_mlp(128, [(64, 'leaky_relu'), (32, 'sigmoid'), (64, 'tanh')])
        x = torch.randn(32, 128)
        y_hat = net(x)
        self.assertEqual(y_hat.shape, (32, 64))


if __name__ == "__main__":
    unittest.main()
