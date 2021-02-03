from ...data import items, attributes, df, dataloader
import torch

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

hidden_units = 4

model = torch.nn.Sequential(
    ("one", torch.nn.Linear(NUM_ITEMS, hidden_units)),
    ("fone", torch.nn.ReLU()),
    ("two", torch.nn.Linear(hidden_units, NUM_ATTRIBUTES)),
    ("ftwo", torch.nn.ReLU()),
)
