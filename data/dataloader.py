from torch.utils.data import Dataset, DataLoader
from . import items, attributes, df
import torch

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

features = torch.eye(NUM_ITEMS, dtype=torch.float32)
targets = torch.tensor(df.values, dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        super()
        self.features = features
        self.targets = targets
        self.df = df

        assert len(targets) == len(features)
        
        self.n_samples = len(targets)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (self.features[index], self.targets[index])

dataset = CustomDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: batch)