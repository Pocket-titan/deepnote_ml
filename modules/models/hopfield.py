import torch


class Hopfield(torch.nn.Module):
    def __init__(self, hidden_dim: int, recurrent_generations: int):
        super().__init__()
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
