from typing import Tuple, List
from .test import test_model
import pandas as pd
import torch


def reset_model(model: torch.nn.Module) -> None:
    for index, layer in enumerate(model.children()):
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        elif len([x for x in layer.parameters()]) > 0:
            print(f"Failed to reset layer at index: {index}!")


def init_weights(layer) -> None:
    if type(layer) == torch.nn.Linear:
        torch.nn.init.uniform_(layer.weight, 0, 0.9)  # (weight, mean, stdev)
        torch.nn.init.constant_(layer.bias, 0)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.MSELoss,
) -> float:
    model.train()
    loss = 0

    for i, batch in enumerate(dataloader):
        for j, (x, y) in enumerate(batch):
            y_pred = model(x)
            loss += loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.MSELoss,
    epochs: int = 2500,
    test_interval: int = 1,
):
    reset_model(model)
    train_loss = []
    pred_df = pd.DataFrame()
    rep_df = pd.DataFrame()

    for epoch in range(epochs + 1):
        loss = train_epoch(model, optimizer, dataloader, loss_fn)
        train_loss.append(loss)

        if epoch % test_interval == 0:
            predictions, accuracies, losses = test_model(
                model, dataloader, loss_fn)
            # current_pred_df['epoch'] = [epoch] * len(current_pred_df)
            # current_rep_df['epoch'] = [epoch] * len(current_rep_df)
            # pred_df = pd.concat([pred_df, current_pred_df])
            # rep_df = pd.concat([rep_df, current_rep_df])

    return train_loss, (pred_df, rep_df)
