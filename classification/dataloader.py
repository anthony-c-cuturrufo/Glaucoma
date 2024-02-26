import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from classification.dataloader_utils import *

class ScanDataset(Dataset):
    def __init__(self, data, targets, transform, add_denoise=False, contrastive_mode='None', num_denoised=0):
        super(ScanDataset, self).__init__()

        self.data = (torch.tensor(data[0], dtype=torch.float32), torch.tensor(data[1], dtype=torch.float32)) if contrastive_mode == "MacOp" else torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.transform = transform 
        self.contrastive_mode = contrastive_mode
        self.add_denoise = add_denoise
        self.num_denoised = num_denoised

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.contrastive_mode == "None":
            if self.add_denoise and index > (len(self.targets) - self.num_denoised):
                data_point = {
                    "data": self.data[index],
                    "target": self.targets[index],
                }
            else:
                data_point = {
                    "data": self.transform(self.data[index]) if self.transform else self.data[index],
                    "target": self.targets[index],
                }
            return data_point 
        elif self.contrastive_mode == "augmentation":
            data = self.data[index]
            aux  = self.data[index]
            if self.transform:
                data = self.transform(data)
                aux = self.transform(aux)
            data_point = {
                "data": data,
                "aux" : aux,
                "target": self.targets[index],
            }
            aux_point = {
                "data": aux,
                "target": self.targets[index],
            }

            return data_point,aux_point
        else:
            data = self.data[0][index]
            aux  = self.data[1][index]
            if self.transform:
                data = self.transform(data)
                aux = self.transform(aux)
            
            data_point = {
                "data": data,
                "aux" : aux,
                "target": self.targets[index],
            }
            return data_point

