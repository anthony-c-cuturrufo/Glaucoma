import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from monai.transforms import (
    Compose,  
    RandGaussianNoise,
    RandScaleIntensity,
    RandAdjustContrast,
    RandAffine,
)
from torch.utils.data import DataLoader
from classification.dataloader import OCTDataset, OCTDataset_MacOp, ScanDataset
from classification.model_factory import model_factory, ContrastiveLoss, FocalLoss
from classification.dataloader_utils import process_scans, precompute_dataset
import argparse
import pandas as pd

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

dist.init_process_group(backend='nccl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on OCTDataset')
    parser.add_argument('--model_name', type=str, default="ResNext50", help='Name of the model to use (e.g., ResNext50, ViT, 3DCNN, and ResNext121)')
    parser.add_argument('--cuda', type=str, default="cuda", help='CUDA device to use (e.g., cuda:0, cuda:1, etc.)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--contrastive_loss', type=int, default=1, help='1 for contrastive loss, 0 for no contrastive loss.')
    parser.add_argument('--dropout', type=float, default=.2, help='Dropout rate for model')
    parser.add_argument('--contrastive_mode', type=str, default="None", help='Contrastive learning mode (e.g. augmentation or MacOp')
    parser.add_argument('--augment', type=bool, default=True, help='Apply data augmentation')
    parser.add_argument('--precompute', type=bool, default=False, help='Precompute data augmentation')
    parser.add_argument('--test', type=bool, default=False, help='Test training pipeline with only first 10 scans')
    parser.add_argument('--dataset', type=str, default="local_database8_Macular_SubMRN_v4.csv", help='Dataset filename')
    parser.add_argument('--image_size', type=lambda s: tuple(map(int, s.split(','))), default=(128,200,200), help='Image size as a tuple (e.g., 128,200,200)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--add_denoise', type=bool, default=True, help='Add denoised scans (only supported for --precompute option)')
    parser.add_argument('--prob', type=float, default=.5, help='Probability of transformation (e.g .5)')
    parser.add_argument('--imbalance_factor', type=float, default=1.1, help='Multiplicative factor to increase number of glaucoma scans')
    parser.add_argument('--use_focal_loss', type=bool, default=False, help='Use Focal Loss as opposed to Cross Entropy Loss')




    # parser.add_argument('--local_rank', type=int)

    local_rank = int(os.environ['LOCAL_RANK'])

    args = parser.parse_args()
    num_workers = 6
    # device = args.cuda
    device = torch.device(f"cuda:{local_rank}")

    model_name = args.model_name
    batch_size = args.batch_size
    dropout = args.dropout
    contrastive_mode = args.contrastive_mode
    augment_data = args.augment
    dataset_name = args.dataset
    image_size = args.image_size
    use_contrastive_loss = args.contrastive_loss
    num_epochs = args.epochs
    lr = args.lr
    precompute = args.precompute
    weight_decay = args.weight_decay
    test = args.test
    add_denoise = args.add_denoise
    prob = args.prob
    imbalance_factor = args.imbalance_factor
    use_focal_loss = args.use_focal_loss

    print(args)


    # Create Dataset
    #-------------------------------------------------
    print("Creating Dataset")
    df = pd.read_csv(dataset_name)
    transforms = Compose([
        RandGaussianNoise(prob=prob), 
        RandScaleIntensity(prob=prob, factors=(1,4)),
        RandAdjustContrast(prob=prob),
        RandAffine(prob=prob, translate_range=(15,10, 0), rotate_range=(0.02,0,0), scale_range=((-.1, .4), 0,0), padding_mode = "zeros"),
        ])

    if precompute: 
        train_data, val_data, train_targets, val_targets = precompute_dataset(df, transforms, image_size=image_size, contrastive_mode=contrastive_mode, add_denoise=add_denoise, test=test)
        print("Finished processing data")
        train_dataset = ScanDataset(train_data, train_targets, None, contrastive_mode=contrastive_mode)
        val_dataset = ScanDataset(val_data, val_targets, None, contrastive_mode=contrastive_mode)
    else:
        train_data, val_data, train_targets, val_targets = process_scans(df, image_size=image_size, contrastive_mode=contrastive_mode,imbalance_factor=imbalance_factor, test=test)
        print("Finished processing data")
        train_dataset = ScanDataset(train_data, train_targets, transforms, contrastive_mode=contrastive_mode)
        val_dataset = ScanDataset(val_data, val_targets, None, contrastive_mode=contrastive_mode)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    print("Finished creating dataset")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Size of Training Set: ", len(train_targets))
    print("Size of Validation Set: ", len(val_targets))
    #-------------------------------------------------    
    model = model_factory(model_name, image_size, dropout, contrastive_mode=contrastive_mode,device=device).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


    # define the loss function and optimizer
    loss_function = FocalLoss(alpha=0.25, gamma=2) if use_focal_loss else torch.nn.CrossEntropyLoss()
    contrastiveloss = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
    # num_epochs = 500

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        correct_predictions = 0
        total_predictions = 0
        for batch_data in train_dataloader:
            step += 1
            optimizer.zero_grad()
            if contrastive_mode == "None":
                inputs, labels = batch_data['data'].to(device), batch_data['target'].to(device)
                outputs = model(inputs)
                contrastiveloss_value = 0

            else:
                inputs, aux, labels = batch_data[0]['data'].to(device), batch_data[0]['aux'].to(device), batch_data[0]['target'].to(device)
                embedding1,embedding2,outputs = model(inputs,aux)
                contrastiveloss_value = contrastiveloss(embedding1,embedding2, labels)

            loss = loss_function(outputs, labels)

            contrastiveloss_value = contrastiveloss_value if use_contrastive_loss else 0
            loss = loss + contrastiveloss_value if contrastive_mode != "None" else loss
            correct_predictions += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total_predictions += labels.size(0)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataloader) 
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.6f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_accuracy = correct_predictions / total_predictions
        writer.add_scalar("train_accuracy", epoch_accuracy, epoch + 1)


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
                    # pred_probs.extend(F.softmax(val_outputs, dim=1).max(dim=1).values)
                    pred_probs.extend(F.softmax(val_outputs, dim=1)[:, 1].tolist())
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
                # if (epoch + 1) % save_epoch == 0:
                    # torch.save({
                    #     'epoch': (epoch + 1),
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'val_auc': val_metric,
                    #     'train_loss': epoch_loss}, 
                    #     os.path.join(save_dir, str((epoch + 1))+"_classification3d_array.pth"))
    
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
