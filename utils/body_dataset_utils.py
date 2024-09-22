import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SignLanguageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)  # Pad sequences to match the longest one
    labels = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels
