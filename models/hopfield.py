import torch


class HopfieldLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, recurrent_generations: int, F=torch.sigmoid):
        super().__init__()
        self.F = F
        self.recurrent_generations = recurrent_generations
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.uniform_(self.linear.weight, -1, 1)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor):
        for _ in range(self.recurrent_generations):
            y_pred = self.linear.forward(x)
            y_pred = torch.relu(y_pred)
        return x if (self.recurrent_generations == 0) else y_pred
        # edge case when recurrent_generations = 0

    def reset_parameters(self):
        self.linear.reset_parameters()

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class Hopfield(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        recurrent_generations: int,
        F=torch.sigmoid,
    ) -> None:
        super().__init__()
        self.F = F
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.hopfield = HopfieldLayer(hidden_dim, recurrent_generations)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        h = self.F(self.linear1(x))
        rec = self.hopfield(h)
        y_pred = self.F(self.linear2(rec))
        return y_pred

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.hopfield.reset_parameters()
        self.linear2.reset_parameters()
