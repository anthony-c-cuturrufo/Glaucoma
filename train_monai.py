import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import monai
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
from classification.dataloader import OCTDataset
from classification.model_factory import model_factory, ContrastiveLoss
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on OCTDataset')
    parser.add_argument('--model_name', type=str, default="ResNext50", help='Name of the model to use (e.g., ResNext50, ViT, etc.)')
    parser.add_argument('--cuda', type=str, default="cuda:2", help='CUDA device to use (e.g., cuda:0, cuda:1, etc.)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--dropout', type=float, default=.2, help='Dropout rate for model')
    parser.add_argument('--contrastive_mode', type=str, default="None", help='Contrastive learning mode')
    parser.add_argument('--augment', type=bool, default=True, help='Apply data augmentation')
    args = parser.parse_args()

    device = args.cuda
    model_name = args.model_name
    batch_size = args.batch_size
    dropout = args.dropout
    contrastive_mode = args.contrastive_mode
    augment_data = args.augment

    # Create Dataset
    #-------------------------------------------------
    print("Creating Dataset")
    transforms = Compose([
        ToTensor(),
        RandGaussianNoise(), 
        RandScaleIntensity(prob=1, factors=(5,10)),
        RandAdjustContrast(),
        RandAffine(prob=1, translate_range=(15,10, 0), rotate_range=(0.02,0,0), scale_range=((-.1, .4), 0,0), padding_mode = "zeros"),
        ToNumpy(),
        ])
    dataset = OCTDataset(
        "local_database7_Macular_SubMRN_v3.csv", 
        transforms, 
        augment_data = augment_data, 
        contrastive_mode = contrastive_mode)
    print("Done With Dataset")

    #Create Dataloader
    #-------------------------------------------------
    val_patient_ids = dataset.val_patient_ids
    train_patient_ids = dataset.train_patient_ids

    # Use SubsetRandomSampler to create subsets of the dataset
    if contrastive_mode == "None":
        train_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in train_patient_ids])
        val_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i]['patient_id'] in val_patient_ids])
    else:
        train_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i][0]['patient_id'] in train_patient_ids])
        val_sampler = SubsetRandomSampler([i for i in range(len(dataset)) if dataset[i][0]['patient_id'] in val_patient_ids])

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    print("Size of Training Set: ", len(train_dataloader))
    print("Size of Validation Set: ", len(val_dataloader))
    #-------------------------------------------------    
    model = model_factory(model_name, dropout, contrastive_mode=contrastive_mode).to(device)

    # define the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    contrastiveloss = ContrastiveLoss()
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
            optimizer.zero_grad()
            if contrastive_mode == "None":
                inputs, labels = batch_data['data'].to(device), batch_data['target'].to(device)
                outputs = model(inputs)

            else:
                inputs, aux, labels = batch_data[0]['data'].to(device), batch_data[0]['aux'].to(device), batch_data[0]['target'].to(device)
                embedding1,embedding2,outputs = model(inputs,aux)
                contrastiveloss_value = contrastiveloss(embedding1,embedding2, labels)

            loss = loss_function(outputs, labels)
            loss = loss + contrastiveloss_value if contrastive_mode != "None" else loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(dataset) // batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.6f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.6f}")

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

                    if contrastive_mode == "None":
                        val_images, val_labels = val_data['data'].to(device), val_data['target'].cpu()
                        val_outputs = model(val_images)
                    else:
                        val_images,val_aux, val_labels = val_data[0]['data'].to(device), val_data[0]['aux'].to(device), val_data[0]['target'].cpu()
                        val_embedding1,val_embedding2, val_outputs = model(val_images,val_aux)

                    val_outputs = val_outputs.cpu()
                    pred_probs.extend(F.softmax(val_outputs, dim=1).max(dim=1).values)
                    preds.extend(val_outputs.argmax(dim=1))
                    gts.extend(val_labels.argmax(dim=1))
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
                    print("Unknown AUC")
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
