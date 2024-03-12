import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from classification.dataloader_utils import *
import random

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
        elif self.contrastive_mode == "Augmentation":
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
                if self.contrastive_mode == "Denoise":
                    data = self.transform(data)
                else:
                    data = self.transform(data)
                    aux = self.transform(aux)
            data_point = {
                "data": data,
                "aux" : aux,
                "target": self.targets[index],
            }
            return data_point
        
class MRNDataset(Dataset):
    def __init__(self, fp, split_name, split, transform, image_size, add_denoise, contrastive_mode):
        self.split = split
        self.region = "Macular" if "Macular" in fp else "Optic"
        df = pd.read_csv(fp)
        self.df = df[df[split_name].isin(split)].reset_index(drop=True)
        self.data = self.get_data(image_size)
        if add_denoise or contrastive_mode == "Denoise":
            self.denoised_data = self.get_denoised(self.region, image_size)
        else:
            self.denoised_data = None

        self.transform = transform 
        self.contrastive_mode = contrastive_mode
        self.mrn_classification_pairs = self.find_matching_pairs()

    def get_data(self, image_size):
        scans_from_df = lambda fps: [process_scan(adjust_filepath(f), image_size=image_size, left = ("_OS_" in f)) for f in tqdm(fps)]
        return torch.tensor(np.expand_dims(np.array(scans_from_df(self.df.filepaths.values)), axis=1), dtype=torch.float32)
    
    def get_denoised(self, region, image_size):
        base_path = os.path.join("/local2/acc/Glaucoma", "BM3D_data", region + ''.join(map(str, image_size)))
        denoised_images = np.array([np.load(os.path.join(base_path, fp.split("/")[-1][:-4] + ".npy")) for fp in self.df['filepaths'].values])
        denoised_images = torch.tensor(np.expand_dims(denoised_images, axis=1), dtype=torch.float32)
        return denoised_images

    def find_matching_pairs(self):
        return list(zip(self.df['MRN'], self.df['classification']))

    def __len__(self):
        return len(self.mrn_classification_pairs)

    def __getitem__(self, index):
        mrn, classification = self.mrn_classification_pairs[index]
        idx = self.random_select_indices(mrn, classification)
        use_denoised = self.denoised_data is not None and "train" in self.split and random.random() < 0.5 and self.contrastive_mode != "Denoise"
        scan = self.denoised_data[idx] if use_denoised else self.data[idx]

        if self.transform and not use_denoised:
            scan = self.transform(scan)

        if self.contrastive_mode == "None":
            data_point = {
                "data": scan,
                "target": torch.tensor([classification], dtype=torch.float32)
             }
        else:
            data_point = {
                "data": scan,
                "aux": self.denoised_data[idx],
                "target": torch.tensor([classification], dtype=torch.float32)
            }
        return data_point
    
    def random_select_indices(self, mrn, classification):
        indices = self.df.index[(self.df['MRN'] == mrn) & (self.df['classification'] == classification)].tolist()
        idx = random.choice(indices)
        return idx

        
# class MacOpDataset(Dataset):
#     def __init__(self, mc_fp, op_fp, split_name, split, transform, image_size, add_denoise=False):
#         mac_df = pd.read_csv(mc_fp)
#         self.macular_df = mac_df[mac_df[split_name].isin(split)].reset_index(drop=True)
#         self.macular_data = self.get_data(self.macular_df, image_size)

#         op_df = pd.read_csv(op_fp)
#         self.optic_df = op_df[op_df[split_name].isin(split)].reset_index(drop=True)
#         self.optic_data = self.get_data(self.optic_df, image_size)

#         self.mrn_classification_pairs = self.find_matching_pairs()
#         self.transform = transform 

#     def get_data(self, df, image_size):
#         scans_from_df = lambda fps: [process_scan(adjust_filepath(f), image_size=image_size, left = ("_OS_" in f)) for f in tqdm(fps)]
#         return torch.tensor(np.expand_dims(np.array(scans_from_df(df.filepaths.values)), axis=1), dtype=torch.float32)

#     def find_matching_pairs(self):
#         merged_df = pd.merge(self.macular_df, self.optic_df, on=['MRN', 'classification'])
#         return list(zip(merged_df['MRN'], merged_df['classification']))

#     def __len__(self):
#         return len(self.mrn_classification_pairs)

#     def __getitem__(self, index):
#         mrn, classification = self.mrn_classification_pairs[index]

#         macular_idx, optic_idx = self.random_select_indices(mrn, classification)
#         macular_image = self.macular_data[macular_idx]
#         optic_image = self.optic_data[optic_idx]

#         if self.transform:
#             macular_image = self.transform(macular_image)
#             optic_image = self.transform(optic_image)

#         data_point = {
#                 "data": macular_image,
#                 "aux" : optic_image,
#                 "target": torch.tensor([classification]),
#         }

#         return data_point
    
#     def random_select_indices(self, mrn, classification):
#         macular_indices = self.macular_df.index[(self.macular_df['MRN'] == mrn) & (self.macular_df['classification'] == classification)].tolist()
#         optic_indices = self.optic_df.index[(self.optic_df['MRN'] == mrn) & (self.optic_df['classification'] == classification)].tolist()

#         macular_idx = random.choice(macular_indices)
#         optic_idx = random.choice(optic_indices) 
#         return macular_idx, optic_idx

