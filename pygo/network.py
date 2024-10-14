"""neural networks in pytorch"""
import torch
from torch import nn
from typing import Tuple
import numpy as np


class ResBlockAlt(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(self.block(data) + data)


class Conv2d(nn.Module):
    """a variant of `nn.Conv2d` with BatchNorm2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 *, padding: int = 0, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False,
                              stride=stride, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(data))


class PolicyHead(nn.Module):
    """policy head
    """
    def __init__(self, *, board_size: int, channels: int):
        super().__init__()
        last_channels = 1
        self.head = nn.Sequential(
            Conv2d(channels, last_channels, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((board_size ** 2) * last_channels,
                      (board_size ** 2) + 1),  # +1 for pass
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.head(data)


class ValueHead(nn.Module):
    """value head
    """
    def __init__(self, *, board_size: int, channels: int, hidden_layer_size):
        super().__init__()
        self.head = nn.Sequential(
            Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PoolBias(nn.Module):
    """an interpretation of Fig. 12 in the Gumbel MuZero paper"""
    def __init__(self, *, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1x1a = Conv2d(channels, channels, 1)
        self.conv1x1b = Conv2d(channels, channels, 1)
        self.conv1x1out = Conv2d(channels, channels, 1)
        self.linear = nn.Linear(2*channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = nn.functional.relu(self.conv1x1a(x))
        b = nn.functional.relu(self.conv1x1b(x))
        bmax = nn.functional.max_pool2d(b, kernel_size=9)
        bmean = nn.functional.avg_pool2d(b, kernel_size=9)
        b = torch.cat((bmax, bmean),
                      dim=1).squeeze(-1).squeeze(-1)
        b = self.linear(b)
        c = a + b[:, :, None, None]
        return self.conv1x1out(c)


def make_gumbel_az_block(channels: int) -> nn.Module:
    b_channels = channels // 2
    return ResBlockAlt(
        nn.Sequential(
            Conv2d(channels, b_channels, 1),
            nn.ReLU(),
            Conv2d(b_channels, b_channels, 3, padding=1),
            nn.ReLU(),
            Conv2d(b_channels, b_channels, 3, padding=1),
            nn.ReLU(),
            Conv2d(b_channels, channels, 1),
        ))


class BasicBody(nn.Module):
    """Body of networks to provide a good feature vector for heads.
    """
    def __init__(self, *, in_channels: int, channels: int, num_blocks: int,
                 broadcast_every: int = 8):
        super().__init__()
        self.conv1 = Conv2d(in_channels, channels, 3, padding=1)
        self.body = nn.Sequential(
            *[make_gumbel_az_block(channels)
              if (_ + 1) % broadcast_every != 0
              else ResBlockAlt(PoolBias(channels=channels))
              for _ in range(num_blocks)
              ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(x))
        x = self.body(x)
        return x


class PVNetwork(nn.Module):
    """Policy-Value network
    """
    def __init__(self, *, board_size: int,
                 in_channels: int, channels: int,
                 num_blocks: int,
                 value_head_hidden: int = 256, broadcast_every: int = 3):
        super().__init__()
        self.body = BasicBody(in_channels=in_channels, channels=channels,
                              num_blocks=num_blocks,
                              broadcast_every=broadcast_every)
        self.head = PolicyHead(
            board_size=board_size,
            channels=channels,
        )
        self.value_head = ValueHead(
            board_size=board_size,
            channels=channels,
            hidden_layer_size=value_head_hidden
        )
        self.config = {
            'board_size': board_size,
            'in_channels': in_channels,
            'channels': channels,
            'num_blocks': num_blocks,
            'value_head_hidden': value_head_hidden,
            'broadcast_every': broadcast_every,
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """take a batch of input features
        and return a batch of [policies, values].
        """
        x = self.body(x)
        return self.head(x), self.value_head(x)

    @property
    @torch.jit.unused
    def device(self):
        """device where this object is placed"""
        return next(self.parameters()).device

    @staticmethod
    def load(path):
        """load from path"""
        objs = torch.load(path)
        cfg = objs['cfg']
        model = PVNetwork(**cfg)
        if 'model_state_dict' in objs:
            model.load_state_dict(objs['model_state_dict'])
        return (model, cfg)


# Local Variables:
# python-indent-offset: 4
# End:
