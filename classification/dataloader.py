import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from classification.dataloader_utils import process_scan, adjust_filepath
import random
from monai.transforms import NormalizeIntensity
import re

class ScanDataset(Dataset):
    def __init__(self, fp, split_name, split, transform, image_size, add_denoise, contrastive_mode, imbalance_factor):
        self.split = split
        self.region = "Macular" if "Macular" in fp else "Optic"

        if "train" not in split and "15" in fp:
            fp = self.region + "15_og.csv"
            
        temp = pd.read_csv(os.path.join("path", fp))
        temp = temp.sample(frac=1).reset_index(drop=True) if "train" not in split else temp
        self.df = temp[temp[split_name].isin(split)].reset_index(drop=True)

        if imbalance_factor != -1 and "train" in split:
            N = min(len(self.df[self.df.classification == 0]), len(self.df[self.df.classification == 1])) 
            self.df = pd.concat([self.df[self.df.classification == 0], self.df[self.df.classification == 1][:int(N * imbalance_factor)]]).reset_index(drop=True) 
       
        self.data = self.get_data(image_size)
        if add_denoise or contrastive_mode == "Denoise":
            self.denoised_data = self.get_denoised(self.region, image_size)
        else:
            self.denoised_data = None

        self.transform = transform 
        self.contrastive_mode = contrastive_mode
        self.targets = torch.tensor(self.df.classification.values[:, np.newaxis], dtype=torch.float32)

    def get_data(self, image_size):
        scans_from_df = lambda fps: [process_scan(adjust_filepath(f), image_size=image_size, left = ("_OS_" in f)) for f in tqdm(fps)]
        return torch.tensor(np.expand_dims(np.array(scans_from_df(self.df.filepaths.values)), axis=1), dtype=torch.float32)
    
    def get_denoised(self, region, image_size):
        base_path = os.path.join("/local2/acc/Glaucoma", "BM3D_data", region + ''.join(map(str, image_size)))
        denoised_images = np.array([np.load(os.path.join(base_path, os.path.basename(fp)[:-4] + ".npy")) for fp in self.df.filepaths.values])
        denoised_images = torch.tensor(np.expand_dims(denoised_images, axis=1), dtype=torch.float32)
        return denoised_images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        use_denoised = self.denoised_data is not None and "train" in self.split and random.random() < 0.3 and self.contrastive_mode != "Denoise"
        scan = self.denoised_data[idx] if use_denoised else self.data[idx]

        if self.transform and not use_denoised and "train" in self.split:
            scan = self.transform(scan)

        if self.contrastive_mode == "None":
            data_point = {
                "data": scan,
                "target": self.targets[idx]
             }
        else:
            data_point = {
                "data": scan,
                "aux": self.denoised_data[idx],
                "target": self.targets[idx]
            }
        return data_point
        
class MRNDataset(Dataset):
    def __init__(self, fp, split_name, split, transform, image_size, add_denoise, contrastive_mode, imbalance_factor):
        self.split = split
        self.region = "Macular" if "Macular" in fp else "Optic"
        temp = pd.read_csv(os.path.join("path", fp))
        self.df = temp[temp[split_name].isin(split)].reset_index(drop=True)

        if imbalance_factor != -1 and "train" in split:
            unique_mrn_0 = self.df[self.df.classification == 0]['MRN'].unique()
            unique_mrn_1 = self.df[self.df.classification == 1]['MRN'].unique()
            N = min(len(unique_mrn_0), len(unique_mrn_1))
            sampled_mrn_0 = np.random.choice(unique_mrn_0, N, replace=False)
            sampled_mrn_1 = np.random.choice(unique_mrn_1, N, replace=False)
            balanced_df = pd.concat([
                self.df[self.df['MRN'].isin(sampled_mrn_0)],
                self.df[self.df['MRN'].isin(sampled_mrn_1)]
            ]).reset_index(drop=True) 
            self.df = balanced_df

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
        denoised_images = np.array([np.load(os.path.join(base_path, os.path.basename(fp)[:-4] + ".npy")) for fp in self.df.filepaths.values])
        denoised_images = torch.tensor(np.expand_dims(denoised_images, axis=1), dtype=torch.float32)
        return denoised_images

    def find_matching_pairs(self):
        return list(zip(self.df['MRN'], self.df['classification']))

    def __len__(self):
        return len(self.mrn_classification_pairs)

    def __getitem__(self, index):
        mrn, classification = self.mrn_classification_pairs[index]
        idx = self.random_select_indices(mrn, classification)
        use_denoised = self.denoised_data is not None and "train" in self.split and random.random() < 0.3 and self.contrastive_mode != "Denoise"
        scan = self.denoised_data[idx] if use_denoised else self.data[idx]

        if self.transform and not use_denoised and "train" in self.split:
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

        
class MacOpDataset(Dataset):
    def __init__(self, mc_fp, op_fp, split_name, split, transform, image_size, add_denoise):
        self.split = split
        self.add_denoise = add_denoise

        if "train" not in split and "15" in mc_fp:
            mc_fp = "data/Macular15_og.csv"
            op_fp = "data/Optic15_og.csv"

        mac_df = pd.read_csv(mc_fp)
        self.macular_df = mac_df[mac_df[split_name].isin(split)].reset_index(drop=True)
        self.macular_data = self.get_data(self.macular_df, image_size)

        op_df = pd.read_csv(op_fp)
        self.optic_df = op_df[op_df[split_name].isin(split)].reset_index(drop=True)
        self.optic_data = self.get_data(self.optic_df, image_size)

        if self.add_denoise:
            self.denoised_mc = self.get_denoised(self.macular_df, "Macular", image_size)
            self.denoised_op = self.get_denoised(self.optic_df, "Optic", image_size)
        else:
            self.denoised_mc = None
            self.denoised_op = None

        self.mrn_classification_pairs = self.find_matching_pairs()
        self.transform = transform 

    def get_data(self, df, image_size):
        scans_from_df = lambda fps: [process_scan(adjust_filepath(f), image_size=image_size, left = ("_OS_" in f)) for f in tqdm(fps)]
        return torch.tensor(np.expand_dims(np.array(scans_from_df(df.filepaths.values)), axis=1), dtype=torch.float32)

    def find_matching_pairs(self):
        merged_df = pd.merge(self.macular_df, self.optic_df, on=['MRN', 'classification'])
        return list(zip(merged_df['MRN'], merged_df['classification']))
    
    def get_denoised(self, df, region, image_size):
        base_path = os.path.join("/local2/acc/Glaucoma", "BM3D_data", region + ''.join(map(str, image_size)))
        denoised_images = np.array([np.load(os.path.join(base_path, os.path.basename(fp)[:-4] + ".npy")) for fp in df.filepaths.values])
        denoised_images = torch.tensor(np.expand_dims(denoised_images, axis=1), dtype=torch.float32)
        return denoised_images

    def __len__(self):
        return len(self.mrn_classification_pairs)

    def __getitem__(self, index):
        mrn, classification = self.mrn_classification_pairs[index]

        macular_idx, optic_idx = self.random_select_indices(mrn, classification)
        use_denoised = self.add_denoise and "train" in self.split and random.random() < 0.3 
        macular_image = self.denoised_mc[macular_idx] if use_denoised else self.macular_data[macular_idx]
        optic_image = self.denoised_op[optic_idx] if use_denoised else self.optic_data[optic_idx]

        if self.transform and not use_denoised and "train" in self.split:
            macular_image = self.transform(macular_image)
            optic_image = self.transform(optic_image)

        data_point = {
                "data": macular_image,
                "aux" : optic_image,
                "target": torch.tensor([classification], dtype=torch.float32)
        }

        return data_point
    
    def random_select_indices(self, mrn, classification):
        macular_indices = self.macular_df.index[(self.macular_df['MRN'] == mrn) & (self.macular_df['classification'] == classification)].tolist()
        optic_indices = self.optic_df.index[(self.optic_df['MRN'] == mrn) & (self.optic_df['classification'] == classification)].tolist()

        macular_idx = random.choice(macular_indices)
        optic_idx = random.choice(optic_indices) 
        return macular_idx, optic_idx
    

class HiroshiMRN(Dataset):
    def __init__(self, split_name, split, transform, add_denoise, contrastive_mode, imbalance_factor):
        self.split = split
        temp = pd.read_csv('/home/acc/Glaucoma/Glaucoma/data/hiroshi_dataset_splits.csv')
        self.df = temp[temp[split_name].isin(split)].reset_index(drop=True)

        if imbalance_factor != -1 and "train" in split:
            unique_mrn_0 = self.df[self.df.classification == 0]['MRN'].unique()
            unique_mrn_1 = self.df[self.df.classification == 1]['MRN'].unique()
            N = min(len(unique_mrn_0), len(unique_mrn_1))
            sampled_mrn_0 = np.random.choice(unique_mrn_0, N, replace=False)
            sampled_mrn_1 = np.random.choice(unique_mrn_1, N, replace=False)
            balanced_df = pd.concat([
                self.df[self.df['MRN'].isin(sampled_mrn_0)],
                self.df[self.df['MRN'].isin(sampled_mrn_1)]
            ]).reset_index(drop=True) 
            self.df = balanced_df

        
        self.data = self.get_data()
        if add_denoise or contrastive_mode == "Denoise":
            self.denoised_data = self.get_denoised()
        else:
            self.denoised_data = None
        

        self.transform = transform 
        self.contrastive_mode = contrastive_mode
        self.mrn_classification_pairs = self.find_matching_pairs()

    def get_data(self):
        numpy_arrays = [np.load(row['filepaths']).astype(np.float32)[np.newaxis, ...] for _, row in self.df.iterrows()]
        stacked_array = np.transpose(np.stack(numpy_arrays), (0, 1, 3, 4, 2))
        normalize = NormalizeIntensity()
        normalized_array = normalize(stacked_array)
        return torch.tensor(normalized_array).clone().detach()
    
    def get_denoised(self):
        denoised_images = []
        base_path = "/local2/acc/Glaucoma/Hiroshi_ONH_OCT_seg"
        
        for _, row in self.df.iterrows():
            filename = os.path.basename(row['filepaths'])
            denoised_filepath = os.path.join(base_path, filename.replace('.npy', '_seg.npy'))
            if os.path.exists(denoised_filepath):
                denoised_image = np.load(denoised_filepath).astype(np.float32)[np.newaxis, ...]
                denoised_images.append(denoised_image)
            else:
                raise FileNotFoundError(f"Segmented file not found for {filename}")

        stacked_denoised_array = np.transpose(np.stack(denoised_images), (0, 1, 3, 4, 2))
        normalize = NormalizeIntensity()
        normalized_denoised_array = normalize(stacked_denoised_array)
        return torch.from_numpy(normalized_denoised_array.numpy())

    def find_matching_pairs(self):
        return list(zip(self.df['MRN'], self.df['classification']))

    def __len__(self):
        return len(self.mrn_classification_pairs)

    def __getitem__(self, index):
        mrn, classification = self.mrn_classification_pairs[index]
        idx = self.random_select_indices(mrn, classification)
        use_denoised = self.denoised_data is not None and "train" in self.split and random.random() < 0.3 and self.contrastive_mode != "Denoise"
        scan = self.denoised_data[idx] if use_denoised else self.data[idx]

        if self.transform and not use_denoised and "train" in self.split:
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

class HiroshiScan(Dataset):
    def __init__(self, dataset_name, split_name, split, transform, image_size, add_denoise, contrastive_mode, imbalance_factor, post_norm=False, denoise_all=False, denoise_prob=.4, transform_denoise=False):
        if denoise_all: assert add_denoise
        assert not (add_denoise and contrastive_mode=="Denoise")
        assert not (denoise_all and contrastive_mode=="Denoise")

        self.post_norm = post_norm
        self.add_denoise = add_denoise
        self.denoise_all = denoise_all
        self.denoise_prob = denoise_prob
        self.transform_denoise = transform_denoise
        self.split = split
        self.image_size = image_size
        temp = pd.read_csv(f'/home/acc/Glaucoma/Glaucoma/data/{dataset_name}.csv')
        self.df = temp[temp[split_name].isin(split)].reset_index(drop=True)
        self.dataset_name = re.sub(r'\d+|_og', '', dataset_name)


        # if imbalance_factor != -1 and "train" in split:
        #     N = min(len(self.df[self.df.classification == 0]), len(self.df[self.df.classification == 1])) 
        #     self.df = pd.concat([self.df[self.df.classification == 0], self.df[self.df.classification == 1][:int(N * imbalance_factor)]]).reset_index(drop=True) 
       
        self.data = self.get_data()
        if add_denoise or contrastive_mode == "Denoise":
            self.denoised_data = self.get_denoised()
        else:
            self.denoised_data = None

        self.transform = transform 
        self.contrastive_mode = contrastive_mode
        self.targets = torch.tensor(self.df.classification.values[:, np.newaxis], dtype=torch.float32)

    def get_data(self):
        if "Hiroshi" in self.dataset_name:
            numpy_arrays = [np.load(row['filepaths']).astype(np.float32)[np.newaxis, ...] for _, row in self.df.iterrows()]
            stacked_array = np.transpose(np.stack(numpy_arrays), (0, 1, 3, 4, 2))
        else:
            numpy_arrays = [np.load(os.path.join(f"/local2/acc/Glaucoma/{self.dataset_name}/{self.image_size[0]}-{self.image_size[1]}-{self.image_size[2]}/OCT", os.path.basename(row['filepaths']).replace(".img", ".npy")))[np.newaxis, ...] for _, row in self.df.iterrows()]
            stacked_array = np.stack(numpy_arrays)
        if self.post_norm:
            return torch.from_numpy(stacked_array)
        else:
            normalize = NormalizeIntensity(nonzero=True)
            normalized_array = normalize(stacked_array)
            return torch.from_numpy(normalized_array.numpy())

    # Macular & Optic - 1024x200x200 masks are saved as uint8, all other are float32
    def get_denoised(self):
        denoised_images = []

        if "Hiroshi" in self.dataset_name:
            base_path = "/local2/acc/Glaucoma/Hiroshi_ONH_OCT_seg"
            for _, row in self.df.iterrows():
                filename = os.path.basename(row['filepaths'])
                denoised_filepath = os.path.join(base_path, filename.replace('.npy', '_seg.npy'))
                if os.path.exists(denoised_filepath):
                    denoised_image = np.load(denoised_filepath).astype(np.float32)[np.newaxis, ...]
                    denoised_images.append(denoised_image)
                else:
                    raise FileNotFoundError(f"Segmented file not found for {filename}")

            stacked_denoised_array = np.transpose(np.stack(denoised_images), (0, 1, 3, 4, 2))
        else:
            base_path = f"/local2/acc/Glaucoma/{self.dataset_name}/{self.image_size[0]}-{self.image_size[1]}-{self.image_size[2]}/Macular_seg"
            for _, row in self.df.iterrows():
                filename = os.path.basename(row['filepaths'])
                denoised_filepath = os.path.join(base_path, filename.replace('.img', '_seg.npy'))
                if os.path.exists(denoised_filepath):
                    denoised_image = np.load(denoised_filepath).astype(np.float32)[np.newaxis, ...]
                    denoised_images.append(denoised_image)
                else:
                    raise FileNotFoundError(f"Segmented file not found for {filename}")

            stacked_denoised_array = np.stack(denoised_images)

        if self.post_norm:
            return torch.from_numpy(stacked_denoised_array)
        else:
            normalize = NormalizeIntensity(nonzero=True)
            normalized_denoised_array = normalize(stacked_denoised_array)
            return torch.from_numpy(normalized_denoised_array.numpy())
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):            
        use_denoised = (self.add_denoise and "train" in self.split and torch.rand(1).item() < self.denoise_prob) or ("train" not in self.split and self.denoise_all)
        scan = self.denoised_data[idx] if use_denoised else self.data[idx]

        if self.transform:
            if not use_denoised or (use_denoised and self.transform_denoise):
                scan = self.transform(scan)

        if self.contrastive_mode == "None":
            data_point = {
                "data": scan,
                "target": self.targets[idx]
             }
        else:
            data_point = {
                "data": scan,
                "aux": self.transform(self.denoised_data[idx]) if (self.transform and self.transform_denoise) else self.denoised_data[idx],
                "target": self.targets[idx]
            }
        return data_point