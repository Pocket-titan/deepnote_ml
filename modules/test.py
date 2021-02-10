from typing import Tuple, List
import pandas as pd
import torch


def test_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.MSELoss,
) -> Tuple[List, List, List]:
    model.eval()
    predictions, accuracies, losses = [], [], []

    with torch.no_grad():
        for (x, y) in dataloader.dataset:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            accuracy = (y_pred.round() == y).to(dtype=torch.float32).mean()

            predictions.append(y_pred)
            losses.append(loss.detach().item())
            accuracies.append(accuracy.detach().item())

    return (predictions, accuracies, losses)
