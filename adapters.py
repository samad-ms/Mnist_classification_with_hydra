from torch import Tensor, nn, flatten


class LinearAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, flatten_input: bool = False) -> None:
        super().__init__()
        self.flatten_input = flatten_input
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        if self.flatten_input:
            x = flatten(x, 1)
        return self.linear(x)
