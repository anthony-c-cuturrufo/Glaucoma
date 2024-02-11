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

def custom_train_val_split(negs, pos, fixed_count, neg_val_split, pos_val_split):
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

    # for precompute
    # print("remaining_pos.shape: ", len(remaining_pos)) # 478
    # print("remaining_neg.shape: ", len(remaining_negs)) # 49
    # print("overlap_pids.shape: ", len(overlap_pids)) # 0



    # Split remaining MRNs for validation
    def split_ids(id_set, split_ratio):
        id_list = list(id_set)
        np.random.shuffle(id_list)
        split_point = int(len(id_list) * split_ratio)
        return set(id_list[:split_point]), set(id_list[split_point:])

    val_neg, train_neg = split_ids(remaining_negs, neg_val_split)
    val_pos, train_pos = split_ids(remaining_pos, pos_val_split)

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


def split_and_process(df, image_size=(128, 200, 200), imbalance_factor=1.1, add_denoise=False, split_name="split1", region="Macular", split="train"):
    df_split = df[df[split_name] == split]

    N = min(len(df_split[df_split.classification == 0]), len(df_split[df_split.classification == 1])) 
    df_bal = pd.concat([df_split[df_split.classification == 0], df_split[df_split.classification == 1][:int(N * imbalance_factor)]]) if split=="train" else df_split

    scans_from_df = lambda df: [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(df.filepaths.values)]
    data = np.expand_dims(np.array(scans_from_df(df_bal)), axis=1)
    labels = df_bal.classification.values
    targets = np.vstack((1 - labels, labels)).T

    num_denoised = 0 
    if add_denoise and split == "train":
        create_labels = lambda scans, label: np.tile(label, (len(scans), 1)).astype(np.float32)
        if region == "Macular":
            train_patient_ids = np.unique(df_split.MRN.values)
            df_old = pd.read_csv("local_database9_" + region + "_SubMRN_v4.csv")
            old_negs = df_old[(df_old.classification == 0)][1:465].reset_index(drop=True)
            old_pos  = df_old[(df_old.classification == 1)][1:465].reset_index(drop=True)

            print("Loading Denoised Data Augmentations")
            neg_denoising_indices = old_negs[old_negs["MRN"].isin(train_patient_ids)].index
            pos_denoising_indices = old_pos[old_pos["MRN"].isin(train_patient_ids)].index
            denoised_neg = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_neg_464.npy")[:,:,:,neg_denoising_indices]
            denoised_pos = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_pos_464.npy")[:,:,:,pos_denoising_indices]

            print("Transposing Denoised Augmentations")
            denoised_neg_transposed = np.transpose(denoised_neg, (3, 0, 1, 2))
            denoised_neg_transposed = np.expand_dims(denoised_neg_transposed, axis=1)
            denoised_pos_transposed = np.transpose(denoised_pos, (3, 0, 1, 2))
            denoised_pos_transposed = np.expand_dims(denoised_pos_transposed, axis=1)

            data = np.concatenate([data, denoised_pos_transposed, denoised_neg_transposed])
            targets = np.concatenate([targets, create_labels(denoised_pos_transposed, [0, 1]), create_labels(denoised_neg_transposed, [1, 0])])
            print("Added ", len(neg_denoising_indices), "denoised negatives and ", len(pos_denoising_indices), "denoised positives")
            num_denoised = len(neg_denoising_indices) + len(pos_denoising_indices)
        elif region == "Optic":
            assert split_name == "split1"
            denoise_folder = "/local2/acc/Denoised_Glaucoma_Data/Optic"
            
            files = {
                'neg': ['denoised_O_neg_0-100.npy', 'denoised_O_neg_100-200.npy', 'denoised_O_neg_200-300.npy', 'denoised_O_neg_300-400.npy', 'denoised_O_neg_400-500.npy', 'denoised_O_neg_500-650.npy'],
                'pos': ['denoised_O_pos_0-170.npy', 'denoised_O_pos_340-510.npy', 'denoised_O_pos_510-680.npy']
            }

            num_denoised = 0
            for label, file_list in files.items():
                for file_name in file_list:
                    file_path = os.path.join(denoise_folder, file_name)
                    denoised_data = np.load(file_path)
                    denoised_data_transposed = np.transpose(denoised_data, (3, 0, 1, 2))
                    denoised_data_transposed = np.expand_dims(denoised_data_transposed, axis=1)
                    
                    data = np.concatenate([data, denoised_data_transposed])
                    if label == 'neg':
                        targets = np.concatenate([targets, create_labels(denoised_data_transposed, [1, 0])])
                    else:
                        targets = np.concatenate([targets, create_labels(denoised_data_transposed, [0, 1])])
                    num_denoised += len(denoised_data_transposed)
    if split == "train": 
        return data, targets, num_denoised
    return data, targets


def process_scans(df, image_size=(128, 200, 200), contrastive_mode='None', imbalance_factor=1.1, add_denoise=False, test=False, split=None, region="Macular"):
    if split is not None:
        negs = df[(df.classification == 0) & (df[split] != "test")]
        pos  = df[(df.classification == 1) & (df[split] != "test")]
        N = 10 if test else min(len(negs[negs[split] == "train"]), len(negs[negs[split] == "train"])) 
        negs = negs[:N]
        pos = pos[:int(imbalance_factor*N)]
        combined_df = pd.concat([pos, negs], axis=0).reset_index(drop=True)

        # Get indices for train and validation sets
        train_indices = combined_df[combined_df[split] == "train"].index.tolist()
        val_indices = combined_df[combined_df[split] == "val"].index.tolist()

    else:
        negs = df[(df.classification == 0)]
        pos  = df[(df.classification == 1)]
        N = 10 if test else min(len(negs), len(pos)) 

        negs = negs[:N]
        pos = pos[:int(imbalance_factor*N)]

        train_patient_ids, val_patient_ids =  custom_train_val_split(negs, pos, fixed_count=160, neg_val_split=.3, pos_val_split=.2) if add_denoise else train_val_split(negs, pos, val_split=.2) 

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

    if add_denoise:
        if split is not None:
            train_patient_ids = np.unique(df[df[split] == "train"].MRN.values)
            df_old = pd.read_csv("local_database9_" + region + "_SubMRN_v4.csv")
            old_negs = df_old[(df_old.classification == 0)][1:465].reset_index(drop=True)
            old_pos  = df_old[(df_old.classification == 1)][1:465].reset_index(drop=True)

            print("Loading Denoised Data Augmentations")
            neg_denoising_indices = old_negs[old_negs["MRN"].isin(train_patient_ids)].index
            pos_denoising_indices = old_pos[old_pos["MRN"].isin(train_patient_ids)].index
            denoised_neg = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_neg_464.npy")[:,:,:,neg_denoising_indices]
            denoised_pos = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_pos_464.npy")[:,:,:,pos_denoising_indices]


            print("Transposing Denoised Augmentations")
            denoised_neg_transposed = np.transpose(denoised_neg, (3, 0, 1, 2))
            denoised_neg_transposed = np.expand_dims(denoised_neg_transposed, axis=1)

            denoised_pos_transposed = np.transpose(denoised_pos, (3, 0, 1, 2))
            denoised_pos_transposed = np.expand_dims(denoised_pos_transposed, axis=1)

            train_data = np.concatenate([train_data, denoised_pos_transposed, denoised_neg_transposed])
            train_targets = np.concatenate([train_targets, create_labels(denoised_pos_transposed, [0, 1]), create_labels(denoised_neg_transposed, [1, 0])])

            print("Added ", len(neg_denoising_indices), "denoised negatives and ", len(pos_denoising_indices), "denoised positives")


        else:
            print("Loading Denoised Data Augmentations")
            denoised_neg = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_neg_150.npy")
            denoised_pos = np.load("/local2/acc/Denoised_Glaucoma_Data/denoised_pos_150.npy")

            print("Transposing Denoised Augmentations")
            denoised_neg_transposed = np.transpose(denoised_neg, (3, 0, 1, 2))
            denoised_neg_transposed = np.expand_dims(denoised_neg_transposed, axis=1)

            denoised_pos_transposed = np.transpose(denoised_pos, (3, 0, 1, 2))
            denoised_pos_transposed = np.expand_dims(denoised_pos_transposed, axis=1)

            train_data = np.concatenate([train_data, denoised_pos_transposed, denoised_neg_transposed])
            train_targets = np.concatenate([train_targets, create_labels(denoised_pos_transposed, [0, 1]), create_labels(denoised_neg_transposed, [1, 0])])

    if contrastive_mode == "MacOp":
        optic_normal_scans, optic_abnormal_scans = map(scans_from_df, [negs.filepaths_optic, pos.filepaths_optic])
        optic_normal_scans, optic_abnormal_scans = map(add_channel_dim, [optic_normal_scans, optic_abnormal_scans])

        optic_data = np.concatenate((optic_abnormal_scans, optic_normal_scans))
        optic_train_data, optic_val_data = optic_data[train_indices], optic_data[val_indices]

        return (train_data, optic_train_data), (val_data, optic_val_data), train_targets, val_targets

    return train_data, val_data, train_targets, val_targets, len(neg_denoising_indices) + len(pos_denoising_indices)

def apply_augmentation(scans, transform, count):
    augmented_scans = []
    # print("apply_augmentations FLAG 1: Scans.shape: ", scans.shape) # (N x 1 x W x H x D)
    for scan in scans:
        for _ in range(int(count // len(scans))):
            augmented_scan = transform(scan).numpy()
            # print("apply_augmentations FLAG 1: augmented_scan.shape:", augmented_scan.shape) # (1 x W x H x D)

            augmented_scans.append(augmented_scan)
    return np.array(augmented_scans)

def get_test_scans(df, image_size=(128, 200, 200), split=None):
    scans_from_df = lambda df: [process_scan(adjust_filepath(f), image_size=image_size) for f in tqdm(df.values)]
    add_channel_dim = lambda scans: np.expand_dims(np.array(scans), axis=1)
    df_test = df[(df[split] == "test")]
    data = add_channel_dim(scans_from_df(df_test.filepaths))
    labels = df_test.classification.values
    targets = np.vstack((1 - labels, labels)).T
    return data, targets

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
    train_patient_ids, val_patient_ids = custom_train_val_split(negs, pos, fixed_count=160, neg_val_split=0.4, pos_val_split=.2)

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
