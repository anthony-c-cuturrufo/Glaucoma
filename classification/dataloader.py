import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from classification.dataloader_utils import *

class OCTDataset(Dataset):
    def __init__(self, filename, transform, augment_data = True, contrastive_mode = "None"):
        self.transform = transform
        self.contrastive_mode = contrastive_mode
        df = pd.read_csv(filename)
        N = 238
        
        negs = df[(df.classification == 0) & (df.filepaths != "-1")]
        pos = df[(df.classification == 1) & (df.filepaths != "-1")]
 
        normal_scans = [process_scan(adjust_filepath(f)) for f in tqdm(negs.filepaths.values[:N])]
        abnormal_scans = [process_scan(adjust_filepath(f)) for f in tqdm(pos.filepaths.values[:N])]

        unique_pos_pids = list(set(pos.MRN.values[:N]))
        unique_neg_pids = list(set(negs.MRN.values[:N]))

        np.random.shuffle(unique_pos_pids)
        np.random.shuffle(unique_neg_pids)

        val_split = 0.2
        # num_pos_val_patients = int(np.ceil(val_split * len(unique_pos_pids)))
        # num_neg_val_patients = int(np.ceil(val_split * len(unique_neg_pids)))
        num_pos_val_patients = min(int(np.ceil(val_split * len(unique_pos_pids))), int(np.ceil(val_split * len(unique_neg_pids))))
        num_neg_val_patients = num_pos_val_patients

        self.val_patient_ids = np.concatenate((unique_pos_pids[:num_pos_val_patients], unique_neg_pids[:num_neg_val_patients]))
        self.train_patient_ids = np.concatenate((unique_pos_pids[num_pos_val_patients:], unique_neg_pids[num_neg_val_patients:]))

        if augment_data:
            # data augmentations - normals 
            new_scans = [transform(normal_scans[i]) for i in range(len(normal_scans)) if negs.MRN.values[i] in self.train_patient_ids] 
            new_scan_ids = np.array([negs.MRN.values[i] for i in range(len(normal_scans)) if negs.MRN.values[i] in self.train_patient_ids])
            normal_scans = np.array(normal_scans + new_scans)

            # data augmentations - abnormals
            pos_new_scans = [transform(abnormal_scans[i]) for i in range(len(abnormal_scans)) if pos.MRN.values[i] in self.train_patient_ids][:N] 
            pos_new_scan_ids = np.array([pos.MRN.values[i] for i in range(len(abnormal_scans)) if pos.MRN.values[i] in self.train_patient_ids])[:N]
            abnormal_scans = np.array(abnormal_scans + pos_new_scans)

        else:
            normal_scans = np.array(normal_scans)
            abnormal_scans = np.array(abnormal_scans) 

        #create labels        
        normal_labels = np.tile([1, 0], (len(normal_scans), 1)).astype(np.float32)
        abnormal_labels = np.tile([0, 1], (len(abnormal_scans), 1)).astype(np.float32)
        
        self.data = np.concatenate((abnormal_scans, normal_scans), axis=0)
        self.targets = np.concatenate((abnormal_labels, normal_labels), axis=0)
        
        if augment_data: 
            self.patient_ids = np.concatenate((pos.MRN.values[:N], pos_new_scan_ids, negs.MRN.values[:N], new_scan_ids))
        else: 
            self.patient_ids = np.concatenate((pos.MRN.values[:N], negs.MRN.values[:N]))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        if self.contrastive_mode == "None":
            data = torch.tensor(self.data[index])
            # if self.transform:
            #     data = self.transform(data)
            data_point = {
                "data": data,
                "target": torch.tensor(self.targets[index]),
                "patient_id": self.patient_ids[index]
            }
            return data_point 
        else:
            data = torch.tensor(self.data[index])
            aux = torch.tensor(self.data[index])
            if self.transform:
                data = self.transform(data)
                aux = self.transform(aux)
            data_point = {
                "data": data,
                "aux" : aux,
                "target": torch.tensor(self.targets[index]),
                "patient_id": self.patient_ids[index]
            }
            aux_point = {
                "data": aux,
                "target": torch.tensor(self.targets[index]),
                "patient_id": self.patient_ids[index]
            }

            return data_point,aux_point
    
    