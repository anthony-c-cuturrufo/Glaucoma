import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from classification.dataloader_utils import *

class OCTDataset(Dataset):
    def __init__(self, filename, transform, augment_data = True, contrastive_mode = "None", image_size = (128, 512, 64)):
        self.transform = transform
        self.contrastive_mode = contrastive_mode
        df = pd.read_csv(filename)
        N = len(df[(df.classification == 0) & (df.filepaths != "-1")])
        
        negs = df[(df.classification == 0) & (df.filepaths != "-1")]
        pos = df[(df.classification == 1) & (df.filepaths != "-1")]
 
        normal_scans = [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(negs.filepaths.values[:N])]
        abnormal_scans = [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(pos.filepaths.values[:N])]

        unique_pos_pids = list(set(pos.MRN.values[:N]))
        unique_neg_pids = list(set(negs.MRN.values[:N]))

        rng = np.random.RandomState(42)
        rng.shuffle(unique_pos_pids)
        rng.shuffle(unique_neg_pids)

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

        # add channel dimension
        normal_scans = np.expand_dims(normal_scans, axis=1)
        abnormal_scans = np.expand_dims(abnormal_scans, axis=1)

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


'''
Dataset for Contrastive Learning method "MacOp" what creates a siamese model comparing
the embedding space of a Macular Scan and Optic Scan from the same patient
'''
class OCTDataset_MacOp(Dataset):
    def __init__(self, filename, transform, augment_data = False, image_size = (128, 512, 64)):
        self.transform = transform
        df = pd.read_csv(filename)
        
        N = len(df[(df.classification == 0)])
        augment_data = False 
        
        negs = df[(df.classification == 0)]
        pos = df[(df.classification == 1)]
 
        macular_normal_scans = [process_scan(adjust_filepath(f),image_size=image_size) for f in tqdm(negs.filepaths_macular.values[:N])]
        macular_abnormal_scans = [process_scan(adjust_filepath(f),image_size=image_size) for f in tqdm(pos.filepaths_macular.values[:N])]
        optic_normal_scans = [process_scan(adjust_filepath(f),image_size=image_size) for f in tqdm(negs.filepaths_optic.values[:N])]
        optic_abnormal_scans = [process_scan(adjust_filepath(f),image_size=image_size) for f in tqdm(pos.filepaths_optic.values[:N])]

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
            # TODO NOT SUPPORTED
            # data augmentations 
            new_scans = [transform(normal_scans[i]) for i in range(len(normal_scans)) if negs.MRN.values[i] in self.train_patient_ids] 
            # new_scans += [transform(normal_scans[i]) for i in range(len(normal_scans)) if negs.STUDY_ID.values[i] in self.train_patient_ids]
            new_scan_ids = np.array([negs.MRN.values[i] for i in range(len(normal_scans)) if negs.MRN.values[i] in self.train_patient_ids])

            # add new scans
            normal_scans = np.array(normal_scans + new_scans)
        else:
            macular_normal_scans = np.array(macular_normal_scans)
            optic_normal_scans = np.array(optic_normal_scans)


        macular_abnormal_scans = np.array(macular_abnormal_scans) 
        optic_abnormal_scans = np.array(optic_abnormal_scans) 

        # add channel dimension
        macular_normal_scans = np.expand_dims(macular_normal_scans, axis=1)
        macular_abnormal_scans = np.expand_dims(macular_abnormal_scans, axis=1)
        optic_normal_scans = np.expand_dims(optic_normal_scans, axis=1)
        optic_abnormal_scans = np.expand_dims(optic_abnormal_scans, axis=1)

        #create labels        
        normal_labels = np.tile([1, 0], (len(macular_normal_scans), 1)).astype(np.float32)
        abnormal_labels = np.tile([0, 1], (len(macular_abnormal_scans), 1)).astype(np.float32)
        
        self.macular_data = np.concatenate((macular_abnormal_scans, macular_normal_scans), axis=0)
        self.optic_data = np.concatenate((optic_abnormal_scans, optic_normal_scans), axis=0)

        self.targets = np.concatenate((abnormal_labels, normal_labels), axis=0)
        
        if augment_data: 
            #TODO NOT SUPPORTED
            self.patient_ids = np.concatenate((pos.MRN.values[:N], negs.MRN.values[:N], new_scan_ids))
        else: 
            self.patient_ids = np.concatenate((pos.MRN.values[:N], negs.MRN.values[:N]))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        data = torch.tensor(self.macular_data[index])
        aux = torch.tensor(self.optic_data[index])
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