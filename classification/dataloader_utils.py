import numpy as np
from scipy import ndimage
from tqdm import tqdm
import pathlib
import pandas as pd
import os

'''
Converts posix filepath to a linux filepath and prepends '/Volumes/Cirrus_Export'
'''
def convert_posix(fp):
    p = pathlib.PureWindowsPath(fp)
    new_p = pathlib.Path('/Volumes/Cirrus_Export').joinpath(*p.parts[1:])
    return str(new_p)

'''
Prepends local2 directory to filepaths in the csv
'''
def adjust_filepath(fp):
    return '/local2/acc/Glaucoma/dataset' + fp 

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

def resize_volume(img, image_size):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = image_size[2]
    desired_width = image_size[0]
    desired_height = image_size[1]
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

def process_scan(path, image_size):
    img = read_image(path)
    img = normalize(img)
    return resize_volume(img, image_size) 

def split_and_process(df, image_size=(128, 200, 200), imbalance_factor=1.1, add_denoise=False, split_name="split1", region="Macular", split=["train"], contrastive_mode = "None"):        
    df_split = df[df[split_name].isin(split)] 
    N = min(len(df_split[df_split.classification == 0]), len(df_split[df_split.classification == 1])) 
    df_bal = pd.concat([df_split[df_split.classification == 0], df_split[df_split.classification == 1][:int(N * imbalance_factor)]]) if "train" in split else df_split
    labels = df_bal.classification.values
    targets = np.vstack((1 - labels, labels)).T
    scans_from_df = lambda fps: [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(fps)]

    if region == "Macop":
        op_data = np.expand_dims(np.array(scans_from_df(df_bal.filepaths.values)), axis=1)
        mc_data = np.expand_dims(np.array(scans_from_df(df_bal.filepaths_mc.values)), axis=1)
        return ((op_data, mc_data), targets, 0) if "train" in split else ((op_data, mc_data), targets)

    data = np.expand_dims(np.array(scans_from_df(df_bal.filepaths.values)), axis=1)
    num_denoised = 0 
    if (add_denoise and "train" in split) or contrastive_mode == "Denoise":
        assert image_size == (128,200,200)
        print("Loading denoised images")
        base_path = os.path.join("/local2/acc/Glaucoma", "BM3D_data", region + ''.join(map(str, image_size)))
        denoised_images = np.array([np.load(os.path.join(base_path, fp.split("/")[-1][:-4] + ".npy")) for fp in df_bal['filepaths'].values])
        denoised_images = np.expand_dims(denoised_images, axis=1)
        print("Adding denoised images")
        if contrastive_mode == "None":
            data = np.concatenate([data, denoised_images])
            targets = np.concatenate([targets, targets])
            num_denoised = len(df_bal.filepaths)
        else:
            assert contrastive_mode == "Denoise"
            return ((data, denoised_images), targets, 0) if "train" in split else ((data, denoised_images), targets) 
       
    if "train" in split: 
        return data, targets, num_denoised
    return data, targets


