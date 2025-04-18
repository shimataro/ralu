import torch


class RaLU(torch.nn.Module):
    """ Rational Linear Unit (RaLU) activation function for PyTorch

    y = x * (x^2 + a) / (x^2 + 1)
    """

    def __init__(self, init_a: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(init_a, dtype=torch.float32, device=device))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward pass

        :param torch.Tensor x: Input tensor.
        :return: Output tensor after applying RaLU activation.
        :rtype: torch.Tensor
        """
        x2 = torch.square(x)
        return x * (x2 + self.a) / (x2 + 1)
