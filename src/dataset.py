import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HCGDataset(Dataset):
    def __init__(self, data_dir, seq_length=900):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.seq_len = seq_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = np.load(self.files[idx])  # (C, T)
        # 如果长度不够，补零；太长，截断
        C, T = mat.shape
        if T < self.seq_len:
            pad = np.zeros((C, self.seq_len - T))
            mat = np.concatenate([mat, pad], axis=1)
        else:
            mat = mat[:, :self.seq_len]
        return torch.FloatTensor(mat)  # 这里暂时没标签，后续你加上浓度和风险标签