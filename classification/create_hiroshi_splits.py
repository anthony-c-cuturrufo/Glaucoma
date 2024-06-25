import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Define the folder path
folder_path = "/local2/acc/Glaucoma/Hiroshi_ONH_OCT"

# List all files in the directory
files = os.listdir(folder_path)

# Initialize an empty list to store the data
data = []

# Process each file
for file in files:
    if file.endswith(".npy"):
        # Split the filename into components
        parts = file.split('-')
        classification = 0 if parts[0] == "Normal" else 1
        mrn = parts[1]
        date = parts[2] #TODO join parts[2:4] to complete date
        eye = parts[5].replace('.npy', '')
        
        # Create the file path
        filepath = os.path.join(folder_path, file)
        
        # Append the data
        data.append([mrn, date, filepath, classification, eye])

# Create a DataFrame
df = pd.DataFrame(data, columns=['MRN', 'Date', 'filepaths', 'classification', 'eye'])

# Create train-test split by patient ID (MRN) without stratification
unique_mrns = df['MRN'].unique()
train_mrns, test_mrns = train_test_split(unique_mrns, test_size=0.1, random_state=42)

# Mark the split column for train and test sets
train_df = df[df['MRN'].isin(train_mrns)].copy()
test_df = df[df['MRN'].isin(test_mrns)].copy()

# Initialize split columns
for i in range(1, 11):
    train_df[f'split{i}'] = 'train'
    test_df[f'split{i}'] = 'test'

# Create KFold object for 10 splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Populate split columns with "val" for validation sets
for i, (train_index, val_index) in enumerate(kf.split(train_df), 1):
    train_df.iloc[val_index, train_df.columns.get_loc(f'split{i}')] = 'val'

# Combine train and test DataFrames
df = pd.concat([train_df, test_df])

# Save the DataFrame to a CSV file
output_file = "hiroshi_dataset_splits.csv"
df.to_csv(output_file, index=False)

print(f"DataFrame with splits saved to {output_file}")
