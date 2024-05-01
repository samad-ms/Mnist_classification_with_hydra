from torch import nn, Tensor


class IdentityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return x

