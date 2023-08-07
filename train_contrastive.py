from tqdm import tqdm  
import logging
import os
import sys
import tempfile

# import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.networks.nets import AHNet
from monai.networks.nets import ResNet

from monai.apps import download_and_extract
from monai.config import print_config
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from monai.data import CacheDataset, DataLoader#, NiftiDataset, Dataset
from monai.transforms import (
    AddChannel,
    AddChanneld,
    Compose, 
    RandRotate90, 
    Resize,
    Resized,
    ScaleIntensity, 
    ToTensor,
    Randomizable,
    # LoadNifti
    ToTensord,
    MapTransform,
    RandGaussianNoise,
    RandScaleIntensity,
    RandAdjustContrast,
    RandAffine,
    ToTensor,
    ToNumpy

)
# from torchvision import transforms


import numpy as np 
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


# from torchvision import transforms
from tqdm import tqdm

from classification.dataloader_contrastive import OCTDataset
from classification.model_contrastive import ResNet, ContrastiveLoss

if __name__ == "__main__":
    print("Creating Dataset")
    transforms = Compose([
        ToTensor(),
        RandGaussianNoise(), 
        RandScaleIntensity(prob=1, factors=(5,10)),
        RandAdjustContrast(),
        RandAffine(prob=1, translate_range=(30,10, 0), rotate_range=(0,.5,0), scale_range=(-.5,.5,0), padding_mode = "zeros"),
        ToNumpy(),
        ])
    dataset = OCTDataset("local_database4.csv", transforms)
    # dataset = torch.load("/local2/acc/Glaucoma/Custom_Dataset/custom_dataset2.pt")
    print("Done With Dataset")

    #Create Dataloader
    #-------------------------------------------------
    val_patient_ids = dataset.val_patient_ids
    train_patient_ids = dataset.train_patient_ids

    # Use SubsetRandomSampler to create subsets of the dataset
    train_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i][0]['patient_id'] in train_patient_ids])
    val_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i][0]['patient_id'] in val_patient_ids])

    batch_size = 3
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    print("Size of Training Set: ", len(train_dataloader))
    print("Size of Validation Set: ", len(val_dataloader))
    #-------------------------------------------------
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:2"
    print(device)
    # model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    # define the input image size and number of output classes
    # input_shape = (1, 64, 64, 64)

    # create the ResNet model
    model = ResNet(spatial_dims=3, in_channels=1, num_classes=1024).to(device)
    # model.class_layers.add_module("sig", torch.nn.Sigmoid())


    # define the loss function and optimizer
    loss_function = torch.nn.BCEWithLogitsLoss()
    contrastiveloss = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # model.class_layers.add_module("sig", torch.nn.Sigmoid())
    # loss_function = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-5)

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
    num_epochs = 400

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader:
            step += 1
            inputs, aux, labels = batch_data[0]['data'].to(device), batch_data[0]['aux'].to(device), batch_data[0]['target'].to(device)
            inputs = inputs.float().unsqueeze(1)
            aux = aux.float().unsqueeze(1)

            optimizer.zero_grad()
            embedding1,embedding2,outputs = model(inputs,aux)

            loss = loss_function(outputs.squeeze().float(), labels.squeeze().float())
            contrastiveloss_value = contrastiveloss(embedding1,embedding2, labels.squeeze().float())
            loss = loss + contrastiveloss_value

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
                    val_images,val_aux, val_labels = val_data[0]['data'].to(device), val_data[0]['aux'].to(device), val_data[0]['target'].cpu()
                    val_images = val_images.float().unsqueeze(1)
                    val_aux = val_aux.float().unsqueeze(1)

                    val_embedding1,val_embedding2, val_outputs = model(val_images,val_aux)
                    val_outputs = val_outputs.cpu().squeeze()
                    # val_outputs = torch.sigmoid(val_outputs[:,1])
                    # print(val_outputs, val_outputs > .5)
                    # print(val_outputs, val_outputs > 0.5, val_labels)
                    pred_probs.extend(val_outputs)
                    preds.extend(val_outputs > 0.5)
                    gts.extend(val_labels)
                    # print(preds, gts)
                acc = accuracy_score(gts, preds)
                f1 = f1_score(gts, preds)
                recall = recall_score(gts, preds)
                prec = precision_score(gts, preds)
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