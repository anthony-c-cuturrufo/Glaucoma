import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from monai.transforms import (
    Compose,  
    ToTensor,
    RandGaussianNoise,
    RandScaleIntensity,
    RandAdjustContrast,
    RandAffine,
    ToTensor,
    ToNumpy

)

from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.metrics import accuracy_score
from classification.dataloader import OCTDataset
from classification.model import Custom3DCNN

if __name__ == "__main__":
    print("Creating Dataset")
    transforms = Compose([
        ToTensor(),
        RandGaussianNoise(), 
        RandScaleIntensity(prob=1, factors=(5,10)),
        RandAdjustContrast(),
        RandAffine(prob=1, translate_range=(15,10, 0), rotate_range=(0.02,0,0), scale_range=((-.1, .4), 0,0), padding_mode = "zeros"),
        ToNumpy(),
        ])
    dataset = OCTDataset("local_database7_Macular_SubMRN_v3.csv", transforms)
    print("Done With Dataset")

    #Create Dataloader
    #-------------------------------------------------
    val_patient_ids = dataset.val_patient_ids
    train_patient_ids = dataset.train_patient_ids

    # Use SubsetRandomSampler to create subsets of the dataset
    train_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in train_patient_ids])
    val_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in val_patient_ids])

    batch_size = 5
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    print("Size of Training Set: ", len(train_dataloader))
    print("Size of Validation Set: ", len(val_dataloader))
    #-------------------------------------------------
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:3"
    print(device)

    # create the ResNet model
    model = monai.networks.nets.SEResNext101(
        spatial_dims=3,
        in_channels=1,
        num_classes=1,
    ).to(device)




    # define the loss function and optimizer
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    val_interval = 1
    save_epoch = 1
    save_dir = "exp2"

    # save_dir = os.path.join(root_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # start a typical PyTorch training
    start_epoch = 0
    num_epochs = 300

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader:
            step += 1
            inputs, labels = batch_data['data'].to(device), batch_data['target'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(dataset) // batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                preds = []
                pred_probs = []
                gts = []
                step = 0
                for val_data in val_dataloader:
                    step += 1
                    val_images, val_labels = val_data['data'].to(device), val_data['target'].cpu()
                    val_outputs = model(val_images).cpu().squeeze(dim=1)
                    pred_probs.extend(torch.sigmoid(val_outputs))
                    preds.extend(val_outputs > 0.5)
                    gts.extend(val_labels)
                acc = accuracy_score(gts, preds)
                f1 = f1_score(gts, preds)
                recall = recall_score(gts, preds)
                prec = precision_score(gts, preds, zero_division=1)
                tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0, 1]).ravel()

                try:
                    spec = float(tn)/(float(tn) + float(fp))
                except:
                    spec = 0
                
                try: 
                    auc = roc_auc_score(gts, pred_probs, labels=[0,1])
                except:
                    auc = 0
                val_metric = auc
                
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    torch.save({
                        'epoch': best_metric_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_metric,
                        'train_loss': epoch_loss}, 
                        os.path.join(save_dir, "best_metric_model_classification3d_array.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current auc_roc: {:.4f} current acc: {:.4f} current f1: {:.4f} \n current recall: {:.4f} current prec: {:.4f} current spec: {:.4f} \n current tp: {:.4f} current fp: {:.4f} current tn: {:.4f} current fn: {:.4f} \n best auc_roc: {:.4f} at epoch {}".format(
                        epoch + 1, auc, acc, f1, recall, prec, spec, tp, fp, tn, fn, best_metric, best_metric_epoch
                    )
                )
                if (epoch + 1) % save_epoch == 0:
                    torch.save({
                        'epoch': (epoch + 1),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_metric,
                        'train_loss': epoch_loss}, 
                        os.path.join(save_dir, str((epoch + 1))+"_classification3d_array.pth"))
    
                writer.add_scalar("val_auc_roc", auc, epoch + 1)
                writer.add_scalar("val_acc", acc, epoch + 1)
                writer.add_scalar("val_f1", f1, epoch + 1)
                writer.add_scalar("val_recall", recall,  epoch + 1)
                writer.add_scalar("val_prec", prec, epoch + 1)
                writer.add_scalar("val_spec", spec, epoch + 1)
                writer.add_scalar("val_tp", tp, epoch + 1)
                writer.add_scalar("val_fp", fp, epoch + 1)
                writer.add_scalar("val_tn", tn, epoch + 1)
                writer.add_scalar("val_fn", fn, epoch + 1)
    print(f"train completed, best_auc: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
