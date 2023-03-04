import numpy as np 
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import transforms
from tqdm import tqdm

from classification.dataloader import OCTDataset
from classification.model import Custom3DCNN

if __name__ == "__main__":
    print("Creating Dataset")
    transforms = torch.nn.Sequential(
    transforms.RandomRotation(45),
)
    dataset = OCTDataset("/Users/anthonycuturrufo/Desktop/1374_patients_path.csv", transforms)
    print("Done With Dataset")

    #Create Dataloader
    #-------------------------------------------------
    # Shuffle list of unique patient IDs in the dataset
    unique_patient_ids = list(set(dataset.patient_ids))
    np.random.shuffle(unique_patient_ids)

    # Split the list of patient IDs into training and validation sets
    val_split = 0.2
    num_val_patients = int(np.ceil(val_split * len(unique_patient_ids)))
    val_patient_ids = unique_patient_ids[:num_val_patients]
    train_patient_ids = unique_patient_ids[num_val_patients:]

    # Use SubsetRandomSampler to create subsets of the dataset
    train_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in train_patient_ids])
    val_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in val_patient_ids])

    batch_size = 4
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    print("Size of Training Set: ", len(train_dataloader))
    print("Size of Validation Set: ", len(val_dataloader))
    #-------------------------------------------------
    
    # Set the hyperparameters
    learning_rate = 0.001
    num_epochs = 150
    num_classes = 1

    # Initialize the model and the optimizer
    model = Custom3DCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val = 0

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        # Training loop
        model.train()
        for batch in train_dataloader:
            data = batch['data'].float()
            targets = batch['target']
            optimizer.zero_grad()
            outputs = model(data)
            loss = nn.BCELoss()(outputs.float().squeeze(), targets.float().squeeze())
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for batch in val_dataloader:
                data = batch['data'].float()
                targets = batch['target']
                outputs = model(data)
                val_loss += nn.BCELoss()(outputs.float().squeeze(), targets.float().squeeze()).item()
                val_acc += accuracy_score(targets, torch.round(outputs))
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

            if val_acc > best_val:
                best_val = val_acc
                print("Saving")
                torch.save(model.state_dict(), "BIG_TEST.pth")

        # Print the epoch and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}")

