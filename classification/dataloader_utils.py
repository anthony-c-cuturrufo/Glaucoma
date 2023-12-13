import numpy as np
from scipy import ndimage
from tqdm import tqdm
import pathlib
import pandas as pd

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
    return img #np.expand_dims(img, axis=0)

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

def custom_train_val_split(negs, pos, fixed_count, val_split):
    np.random.seed(42)  # Ensures reproducibility

    # Select fixed MRNs for training set
    fixed_train_negs = set(negs[:fixed_count].MRN.unique())
    fixed_train_pos = set(pos[:fixed_count].MRN.unique())

    # Combine fixed train MRNs
    fixed_train_mrns = fixed_train_negs.union(fixed_train_pos)

    # Remaining MRNs
    remaining_negs = set(negs.MRN.unique()) - fixed_train_mrns
    remaining_pos = set(pos.MRN.unique()) - fixed_train_mrns

    overlap_pids = remaining_pos.intersection(remaining_negs)
    remaining_pos -= overlap_pids
    remaining_negs -= overlap_pids

    # print("remaining_pos.shape: ", len(remaining_pos)) # 478
    # print("remaining_neg.shape: ", len(remaining_negs)) # 49
    # print("overlap_pids.shape: ", len(overlap_pids)) # 0



    # Split remaining MRNs for validation
    def split_ids(id_set, split_ratio):
        id_list = list(id_set)
        np.random.shuffle(id_list)
        split_point = int(len(id_list) * split_ratio)
        return set(id_list[:split_point]), set(id_list[split_point:])

    val_neg, train_neg = split_ids(remaining_negs, val_split*2)
    val_pos, train_pos = split_ids(remaining_pos, val_split)

     # Split overlap IDs
    half_overlap = len(overlap_pids) // 2
    overlap_list = list(overlap_pids)
    np.random.shuffle(overlap_list)

    train_patient_ids = train_pos.union(train_neg, set(overlap_list[:half_overlap]), fixed_train_mrns)
    val_patient_ids = val_pos.union(val_neg, set(overlap_list[half_overlap:]))

    return train_patient_ids, val_patient_ids


'''
Returns negative and postive evenly split training and validation pids. Deals
with overlapping ids with both pos and neg classification by dividing them in half
'''
def train_val_split(pos, neg, val_split=0.2):
    # Ensure reproducibility
    np.random.seed(42)

    # Identify unique IDs and overlaps
    unique_pos_pids = set(pos.MRN.unique())
    unique_neg_pids = set(neg.MRN.unique())
    overlap_pids = unique_pos_pids.intersection(unique_neg_pids)
    unique_pos_pids -= overlap_pids
    unique_neg_pids -= overlap_pids

    # Split function for unique IDs
    def split_ids(id_set, split_ratio):
        id_list = list(id_set)
        np.random.shuffle(id_list)
        split_point = int(len(id_list) * split_ratio)
        return set(id_list[:split_point]), set(id_list[split_point:])

    train_pos, val_pos = split_ids(unique_pos_pids, 1-val_split)
    train_neg, val_neg = split_ids(unique_neg_pids, 1-val_split)

    # Split overlap IDs
    half_overlap = len(overlap_pids) // 2
    overlap_list = list(overlap_pids)
    np.random.shuffle(overlap_list)
    
    # Combine
    train_patient_ids = train_pos.union(train_neg, set(overlap_list[:half_overlap]))
    val_patient_ids = val_pos.union(val_neg, set(overlap_list[half_overlap:]))

    return train_patient_ids, val_patient_ids

def process_scans(df, image_size=(128, 200, 200), contrastive_mode='None', imbalance_factor=1.1, test=False):
    negs = df[(df.classification == 0)]
    pos = df[(df.classification == 1) ]
    N = 10 if test else min(len(negs), len(pos)) 

    negs = negs[:N]
    pos = pos[:int(imbalance_factor*N)]

    train_patient_ids, val_patient_ids = train_val_split(negs, pos, val_split=.2)

    combined_df = pd.concat([pos, negs], axis=0).reset_index(drop=True)
    
    # Get indices for train and validation sets
    train_indices = combined_df[combined_df['MRN'].isin(train_patient_ids)].index.tolist()
    val_indices = combined_df[combined_df['MRN'].isin(val_patient_ids)].index.tolist()


    scans_from_df = lambda df: [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(df.values)]
    add_channel_dim = lambda scans: np.expand_dims(np.array(scans), axis=1)
    create_labels = lambda scans, label: np.tile(label, (len(scans), 1)).astype(np.float32)

    # Get filepaths based on MacOp
    filepaths = [negs.filepaths_macular, pos.filepaths_macular] if contrastive_mode == "MacOp" else [negs.filepaths, pos.filepaths]
    
    # Process scans and get data
    normal_scans, abnormal_scans = map(scans_from_df, filepaths)
    normal_scans, abnormal_scans = map(add_channel_dim, [normal_scans, abnormal_scans])
    normal_labels, abnormal_labels = map(create_labels, [normal_scans, abnormal_scans], [[1, 0], [0, 1]])

    # Concatenate data and targets
    data = np.concatenate((abnormal_scans, normal_scans))
    targets = np.concatenate((abnormal_labels, normal_labels))

    # Split data into train and validation
    train_data, val_data = data[train_indices], data[val_indices]
    train_targets, val_targets = targets[train_indices], targets[val_indices]

    if contrastive_mode == "MacOp":
        optic_normal_scans, optic_abnormal_scans = map(scans_from_df, [negs.filepaths_optic, pos.filepaths_optic])
        optic_normal_scans, optic_abnormal_scans = map(add_channel_dim, [optic_normal_scans, optic_abnormal_scans])

        optic_data = np.concatenate((optic_abnormal_scans, optic_normal_scans))
        optic_train_data, optic_val_data = optic_data[train_indices], optic_data[val_indices]

        return (train_data, optic_train_data), (val_data, optic_val_data), train_targets, val_targets

    return train_data, val_data, train_targets, val_targets

def apply_augmentation(scans, transform, count):
    augmented_scans = []
    # print("apply_augmentations FLAG 1: Scans.shape: ", scans.shape) # (N x 1 x W x H x D)
    for scan in scans:
        for _ in range(int(count // len(scans))):
            augmented_scan = transform(scan).numpy()
            # print("apply_augmentations FLAG 1: augmented_scan.shape:", augmented_scan.shape) # (1 x W x H x D)

            augmented_scans.append(augmented_scan)
    return np.array(augmented_scans)

def precompute_dataset(df, transforms, image_size=(128, 200, 200), contrastive_mode='None', add_denoise=True, test=False):
    '''
    + 2N   Glaucoma images 
    + 1N   Normal images 
    
    + 2N Glaucoma Augmentation 
    + 3N   Normal Augmentation 

    + 150  Glaucoma Denoising 
    + 150    Normal Denoising 
-------------------------------
    4.5N Glaucoma + 4.5N Normal = 9N Dataset 

    '''
    negs = df[(df.classification == 0)]
    pos = df[(df.classification == 1) ]
    N = 10 if test else min(len(negs), len(pos))

    negs = negs[:N]
    pos = pos[:2*N]

    # Perform train-validation split
    train_patient_ids, val_patient_ids = custom_train_val_split(negs, pos, fixed_count=160, val_split=0.2)

    combined_df = pd.concat([pos, negs], axis=0).reset_index(drop=True)

    print("N: ", N) #465
    print("Train Postive Scans: ",  len(combined_df[(combined_df['classification']==1) & (combined_df['MRN'].isin(train_patient_ids))])) #800
    print("Train Negative Scans: ", len(combined_df[(combined_df['classification']==0) & (combined_df['MRN'].isin(train_patient_ids))])) #391
    print("Val Postitive Scans: ",  len(combined_df[(combined_df['classification']==1) & (combined_df['MRN'].isin(val_patient_ids))])) # 130
    print("Val Negative Scans: ",   len(combined_df[(combined_df['classification']==0) & (combined_df['MRN'].isin(val_patient_ids))])) # 74
    print("Train Postitive MRNs: ", len(combined_df[(combined_df['classification']==1) & (combined_df['MRN'].isin(train_patient_ids))].MRN.unique()))
    print("Train Negative MRNs: ",  len(combined_df[(combined_df['classification']==0) & (combined_df['MRN'].isin(train_patient_ids))].MRN.unique()))
    print("Val Postitive MRNs: ",   len(combined_df[(combined_df['classification']==1) & (combined_df['MRN'].isin(val_patient_ids))].MRN.unique()))
    print("Val Negative MRNs: ",    len(combined_df[(combined_df['classification']==0) & (combined_df['MRN'].isin(val_patient_ids))].MRN.unique()))

    
    # Get indices for train and validation sets
    train_indices = combined_df[combined_df['MRN'].isin(train_patient_ids)].index.tolist()
    val_indices = combined_df[combined_df['MRN'].isin(val_patient_ids)].index.tolist()

     # Get indices for train and validation sets
    pos_train_indices = combined_df[(combined_df['classification']==1) & (combined_df['MRN'].isin(train_patient_ids))].index.tolist()
    neg_train_indices = combined_df[(combined_df['classification']==0) & (combined_df['MRN'].isin(train_patient_ids))].index.tolist()


    scans_from_df = lambda df: [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(df.values)]
    add_channel_dim = lambda scans: np.expand_dims(np.array(scans), axis=1)
    create_labels = lambda scans, label: np.tile(label, (len(scans), 1)).astype(np.float32)

    # Get filepaths based on MacOp
    filepaths = [negs.filepaths_macular, pos.filepaths_macular] if contrastive_mode == "MacOp" else [negs.filepaths, pos.filepaths]
    
    # Process scans and get data
    normal_scans, abnormal_scans = map(scans_from_df, filepaths)
    normal_scans, abnormal_scans = map(add_channel_dim, [normal_scans, abnormal_scans])
    normal_labels, abnormal_labels = map(create_labels, [normal_scans, abnormal_scans], [[1, 0], [0, 1]])

    # Concatenate data and targets
    data = np.concatenate((abnormal_scans, normal_scans))
    targets = np.concatenate((abnormal_labels, normal_labels))

    # Split data into train and validation
    train_data, val_data = data[train_indices], data[val_indices]
    train_targets, val_targets = targets[train_indices], targets[val_indices]

    # Augment positive and negative scans
    print("Creating Positive Data Augmentations")
    augmented_pos_scans = apply_augmentation(data[pos_train_indices], transforms, count=2 * N) # 800 + (2*465) = 1730

    print("Creating Negative Data Augmentations")
    augmented_neg_scans = apply_augmentation(data[neg_train_indices], transforms, count=3 * N) # 391 +(3*465) = 1786

    print("Combining Data Augmentations")
    augmented_train_data = np.concatenate([train_data, augmented_pos_scans, augmented_neg_scans])
    augmented_train_targets = np.concatenate([train_targets, create_labels(augmented_pos_scans, [0, 1]), create_labels(augmented_neg_scans, [1, 0])])
    
    if add_denoise:
        print("Loading Denoised Data Augmentations")
        denoised_neg = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_neg.npy")
        denoised_pos = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_pos.npy")

        print("Transposing Denoised Augmentations")
        denoised_neg_transposed = np.transpose(denoised_neg, (3, 0, 1, 2))
        denoised_neg_transposed = np.expand_dims(denoised_neg_transposed, axis=1)

        denoised_pos_transposed = np.transpose(denoised_pos, (3, 0, 1, 2))
        denoised_pos_transposed = np.expand_dims(denoised_pos_transposed, axis=1)

        augmented_train_data = np.concatenate([augmented_train_data, denoised_pos_transposed, denoised_neg_transposed])
        augmented_train_targets = np.concatenate([augmented_train_targets, create_labels(denoised_pos_transposed, [0, 1]), create_labels(denoised_neg_transposed, [1, 0])])

    # Train data is shape Nx1xWxHxD

    return augmented_train_data, val_data, augmented_train_targets, val_targets
