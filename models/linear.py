import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, F=torch.sigmoid
    ) -> None:
        super().__init__()
        self.F = F
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        y_pred = self.linear1(x)
        y_pred = self.F(y_pred)
        y_pred = self.linear2(y_pred)
        y_pred = self.F(y_pred)
        return y_pred

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
