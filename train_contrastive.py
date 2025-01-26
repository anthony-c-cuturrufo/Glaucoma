import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.transforms import (
    Compose, RandAdjustContrast, RandAffine, RandFlip, NormalizeIntensity, RandGaussianNoise, RandScaleIntensity
)
from classification.dataloader import HiroshiScan  # Import your HiroshiScan dataset
from classification.model_factory import model_factory, SimCLRLoss  # Import model factory and loss

# Training Function
def train_simclr(
    model, criterion, optimizer, scheduler, dataloader, num_epochs=10, device="cuda:3"
):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            data, aux = batch["data"].to(device), batch["aux"].to(device)
            embedding_1, embedding_2 = model(data, aux)

            # Compute contrastive loss
            loss = criterion(embedding_1, embedding_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # Step the scheduler at the end of the epoch
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")


# Main Script
if __name__ == "__main__":
    # Hyperparameters
    dataset_name = "Hiroshi5"  # Replace with your dataset name
    split_name = "split1"  # Replace with your split column name
    train_split = ["train"]
    batch_size = 64
    image_size = (128, 64, 64)  # Adjust to your data dimensions
    contrastive_mode = "Denoise"
    dropout_rate = 0.2
    embedding_dim = 128
    learning_rate = 1e-3
    num_epochs = 15
    model_name = "3DCNN"
    temperature = .5

    # Data Augmentation Transforms
    train_transform = Compose([
        RandGaussianNoise(prob=.4, std=0.01),
        RandScaleIntensity(prob=.4, factors=(0.9, 1.1)),
        RandAdjustContrast(prob=0.4, gamma=(0.9, 1.1)),
        RandAffine(
            prob=0.4, translate_range=(5, 5, 0),
            rotate_range=(0.01, 0, 0), scale_range=((-.1, .1), 0, 0),
            padding_mode="zeros"
        ),
        RandFlip(spatial_axis=0, prob=0.4),
        RandFlip(spatial_axis=1, prob=0.4),
        NormalizeIntensity(nonzero=True)
    ]).set_random_state(seed=1)

    # Dataset and DataLoader
    train_dataset = HiroshiScan(
        dataset_name=dataset_name,
        split_name=split_name,
        split=train_split,
        transform=train_transform,
        image_size=image_size,
        add_denoise=False,
        contrastive_mode=contrastive_mode,
        imbalance_factor=-1,
        post_norm=True,
        denoise_all=False,
        denoise_prob=.5,
        transform_denoise=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model and Loss
    model = model_factory(
        model_name=model_name,
        image_size=image_size,
        dropout=dropout_rate,
        contrastive_mode=contrastive_mode,
        contrastive_layer_size=embedding_dim,
        pretrained=False
    )
    criterion = SimCLRLoss(temperature=temperature)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # Train SimCLR
    train_simclr(model, criterion, optimizer, scheduler, train_loader, num_epochs=num_epochs)

    # Save Pretrained Weights
    os.makedirs("checkpoints", exist_ok=True)
    fpath = f"/local2/acc/Glaucoma/SimCLR/simclr_{model_name}_b{batch_size}_e{num_epochs}_ed{embedding_dim}_lr{learning_rate}_d{dropout_rate}_t{temperature}.pth"
    torch.save(model.state_dict(),fpath)
    print("Pretraining complete")