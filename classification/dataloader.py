import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
from scipy import ndimage
import pathlib
from tqdm import tqdm

'''
Converts posix filepath to a linux filepath and prepends '/Volumes/Cirrus_Export'
'''

def convert_posix(fp):
    p = pathlib.PureWindowsPath(fp)
    new_p = pathlib.Path('/Volumes/Cirrus_Export').joinpath(*p.parts[1:])
    return str(new_p)

'''
Input: Filepath for OCT Scan 
Returns a (1024 (x), 200 (y), 200 (z)) shape image
'''
def read_image(scan_path):
#     print(scan_path)
    Nx=200; Nz=1024; Ny=200;
    if "512x128" in scan_path:
        Nx=128; Nz=1024; Ny=512;
        
    images = []
    with open(scan_path, 'rb') as fid:
        data_array = np.fromfile(fid, np.uint8)
        try:
            data_matrix = data_array.reshape(Nx,Nz,Ny)
        except:
            print(scan_path)
            return 0
        data_matrix = data_matrix.transpose(1,0,2)

    for i in range(Ny):
        data_matrix[:,:,i] = np.fliplr(np.flipud(np.squeeze(data_matrix[:,:,i])))
    return data_matrix

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 256
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def normalize(volume):
    """Normalize the volume"""
    min = 0
    max = 255
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    img = read_image(path)
    img = normalize(img)
    return resize_volume(img) 


class OCTDataset(Dataset):
    def __init__(self, filename, transform):
        self.transform = transform
        df = pd.read_csv(filename)
        
        N = 100
        
        negs = df[df.glaucoma == 0]
        pos = df[df.glaucoma == 1]
 
        normal_scans = np.array([process_scan(convert_posix(f)) for f in tqdm(negs.image_path.values[:N])])
        abnormal_scans = np.array([process_scan(convert_posix(f)) for f in tqdm(pos.image_path.values[:N])]) 
                
        normal_labels = np.array([0 for _ in range(len(normal_scans))])
        abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
        
        self.data = np.concatenate((abnormal_scans, normal_scans), axis=0)
        self.targets = np.concatenate((abnormal_labels, normal_labels), axis=0)
        
        self.patient_ids = np.concatenate((pos.STUDY_ID.values[:N], negs.STUDY_ID.values[:N]))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        if self.transform:
            data = self.transform(data)
        data_point = {
            "data": data,
            "target": torch.tensor(self.targets[index]),
            "patient_id": self.patient_ids[index]
        }
        return data_point 