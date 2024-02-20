import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.transforms import (
    Compose,  
    RandGaussianNoise,
    RandScaleIntensity,
    RandAdjustContrast,
    RandAffine,
)
from torch.utils.data import DataLoader
from classification.dataloader import ScanDataset
from classification.model_factory import model_factory, ContrastiveLoss, FocalLoss
from classification.dataloader_utils import process_scans, precompute_dataset, split_and_process
import argparse
import pandas as pd

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ConfusionMatrix
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


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
    parser.add_argument('--dataset', type=str, default="database11_Macular_SubMRN_v4.csv", help='Dataset filename')
    parser.add_argument('--image_size', type=lambda s: tuple(map(int, s.split(','))), default=(128,200,200), help='Image size as a tuple (e.g., 128,200,200)')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--add_denoise', type=bool, default=False, help='Add denoised scans (only supported for contrastive_mode = None)')
    parser.add_argument('--prob', type=float, default=.5, help='Probability of transformation (e.g .5)')
    parser.add_argument('--imbalance_factor', type=float, default=1.1, help='Multiplicative factor to increase number of glaucoma scans')
    parser.add_argument('--split', type=str, default="split1", help='Train val split (e.g. split1, split2, split3)')
    parser.add_argument('--conv_layers', type=lambda s: [int(item) for item in s.split(',')], default=[32,64], help='Conv layer config for 3DCNN (e.g., 32,64)')
    parser.add_argument('--fc_layers', type=lambda s: [int(item) for item in s.split(',')], default=[16], help='Fc layer config for 3DCNN (e.g., 16)')
    parser.add_argument('--warmup_epochs', type=int, default=-1, help='Number of warmup epochs')
    parser.add_argument('--cos_anneal', type=int, default=-1, help='T_max for cosine annealing lr (e.g 400)')
    parser.add_argument('--loss_f', type=str, default="CrossEntropy", help='Loss function (e.g., CrossEntropy, Focal, CrossEntropyW)')



    local_rank = int(os.environ['LOCAL_RANK'])

    args = parser.parse_args()
    num_workers = 4
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
    weight_decay = args.weight_decay
    add_denoise = args.add_denoise
    prob = args.prob
    imbalance_factor = args.imbalance_factor
    split = args.split
    conv_layers = args.conv_layers
    fc_layers = args.fc_layers
    warmup_epochs = args.warmup_epochs
    cos_anneal = args.cos_anneal
    loss_f = args.loss_f

    num_gpus = dist.get_world_size()
    region = "Macular" if "Macular" in dataset_name else "Optic"

    print(args)


    # Create Dataset
    #--------------------------------------------------------------------------------------------------
    print("Creating Dataset from " + dataset_name)
    df = pd.read_csv(dataset_name)
    transforms = Compose([
        RandGaussianNoise(prob=prob), 
        RandScaleIntensity(prob=prob, factors=(1,4)),
        RandAdjustContrast(prob=prob),
        RandAffine(prob=prob, translate_range=(15,10, 0), rotate_range=(0.02,0,0), scale_range=((-.1, .4), 0,0), padding_mode = "zeros"),
        ])

    # train_data, val_data, train_targets, val_targets, num_denoised = process_scans(df, image_size=image_size, contrastive_mode=contrastive_mode,imbalance_factor=imbalance_factor, add_denoise=add_denoise, test=test, split=split, region=region)
    train_data, train_targets, num_denoised = split_and_process(df, image_size=image_size, imbalance_factor=imbalance_factor, add_denoise=add_denoise, split_name=split, region=region, split="train")
    val_data, val_targets = split_and_process(df, image_size=image_size, imbalance_factor=imbalance_factor, add_denoise=False, split_name=split, region=region, split="val")
    test_data, test_targets = split_and_process(df, image_size=image_size, imbalance_factor=imbalance_factor, add_denoise=False, split_name=split, region=region, split="test")
    print("Finished processing data")
    train_dataset = ScanDataset(train_data, train_targets, transforms, add_denoise=add_denoise, contrastive_mode=contrastive_mode, num_denoised=num_denoised)
    val_dataset = ScanDataset(val_data, val_targets, None, contrastive_mode=contrastive_mode)
    test_dataset = ScanDataset(test_data, test_targets, None, contrastive_mode=contrastive_mode)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)
    print("Finished creating dataset")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,sampler=test_sampler)
    print("Size of Training Set: ", len(train_targets))
    print("Size of Validation Set: ", len(val_targets))
    #-------------------------------------------------    
    model = model_factory(model_name, image_size, dropout, contrastive_mode=contrastive_mode,device=device, conv_layers=conv_layers, fc_layers=fc_layers).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Synchronize all processes
    torch.distributed.barrier() 

    # define the loss function and optimizer
    if loss_f == "Focal":
        loss_function = FocalLoss(alpha=0.25, gamma=2)
    elif loss_f == "CrossEntropyW":
        assert split == "split1" and region == "Optic"
        Class_0_train, Class_0_test, Class_1_train, Class_1_test = 614, 158, 993, 158
        weight_for_Class_0 = (Class_0_test / Class_0_train) * (Class_1_train / Class_1_test)
        weights = torch.tensor([weight_for_Class_0, 1.0], dtype=torch.float32).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    # contrastiveloss = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if cos_anneal != -1:
        scheduler = CosineAnnealingLR(optimizer, T_max = cos_anneal)
    else: 
        lambda1 = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1
        scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    val_interval = 1
    save_epoch = 1
    root_dir = "exp"
    save_dir = region + "_" + split + "_" + model_name + "_" + str(num_gpus) + "_" + str(batch_size) + "_" + str(weight_decay) + "_" + str(imbalance_factor)

    save_dir = os.path.join(root_dir, save_dir)
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    # Train
    #--------------------------------------------------------------------------------------------------
    accuracy = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary').to(device)
    precision = Precision(task='binary').to(device)
    recall = Recall(task='binary').to(device)
    auroc = AUROC(task="binary").to(device)
    confmat = ConfusionMatrix(task="binary").to(device)

    test_accuracy = Accuracy(task='binary').to(device)
    test_f1 = F1Score(task='binary').to(device)
    test_precision = Precision(task='binary').to(device)
    test_recall = Recall(task='binary').to(device)
    test_auroc = AUROC(task="binary").to(device)
    test_confmat = ConfusionMatrix(task="binary").to(device)

    start_epoch = 0
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
                # contrastiveloss_value = 0
            # else:
            #     inputs, aux, labels = batch_data[0]['data'].to(device), batch_data[0]['aux'].to(device), batch_data[0]['target'].to(device)
            #     embedding1,embedding2,outputs = model(inputs,aux)
            #     contrastiveloss_value = contrastiveloss(embedding1,embedding2, labels)

            loss = loss_function(outputs, labels)

            # contrastiveloss_value = contrastiveloss_value if use_contrastive_loss else 0
            # loss = loss + contrastiveloss_value if contrastive_mode != "None" else loss
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
        scheduler.step()
        writer.add_scalar("train_accuracy", epoch_accuracy, epoch + 1)


        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.6f}")

        torch.distributed.barrier() 


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                accuracy.reset()
                f1.reset()
                precision.reset()
                recall.reset()
                auroc.reset()
                confmat.reset()

                step = 0
                for val_data in val_dataloader:
                    step += 1

                    if contrastive_mode == "None":
                        val_images, val_labels = val_data['data'].to(device), val_data['target'].to(device)
                        val_outputs = model(val_images).to(device)
                    # else:
                    #     val_images,val_aux, val_labels = val_data[0]['data'].to(device), val_data[0]['aux'].to(device), val_data[0]['target'].cpu()
                    #     val_embedding1,val_embedding2, val_outputs = model(val_images,val_aux)
                   
                    preds = val_outputs.argmax(dim=1)
                    labels = val_labels.argmax(dim=1)
                    accuracy.update(preds, labels)
                    f1.update(preds, labels)
                    precision.update(preds, labels)
                    recall.update(preds, labels)
                    auroc.update(F.softmax(val_outputs, dim=1)[:, 1], labels)
                    confmat.update(preds, labels)

                torch.distributed.barrier()
                acc = accuracy.compute()
                f1_score = f1.compute()
                prec = precision.compute()
                rec = recall.compute()
                auc = auroc.compute()
                cm = confmat.compute()
                tn, fp, fn, tp = cm.view(-1).tolist()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Reset metrics for the next epoch
                accuracy.reset()
                f1.reset()
                precision.reset()
                recall.reset()
                auroc.reset()
                confmat.reset()
                val_metric = auc

                # TESTING ------------------------------------------------------------------
                model.eval()
                with torch.no_grad():
                    test_accuracy.reset()
                    test_f1.reset()
                    test_precision.reset()
                    test_recall.reset()
                    test_auroc.reset()
                    test_confmat.reset()           
                    for test_data in test_dataloader:
                        if contrastive_mode == "None":
                            test_images, test_labels = test_data['data'].to(device), test_data['target'].to(device)
                            test_outputs = model(test_images).to(device)
            
                        test_preds = test_outputs.argmax(dim=1)
                        test_labels = test_labels.argmax(dim=1)
                        test_accuracy.update(test_preds, test_labels)
                        test_f1.update(test_preds, test_labels)
                        test_precision.update(test_preds, test_labels)
                        test_recall.update(test_preds, test_labels)
                        test_auroc.update(F.softmax(test_outputs, dim=1)[:, 1], test_labels)
                        test_confmat.update(test_preds, test_labels)
                    torch.distributed.barrier()
                    test_acc      = test_accuracy.compute()
                    test_f1_score = test_f1.compute()
                    test_prec     = test_precision.compute()
                    test_rec      = test_recall.compute()
                    test_auc      = test_auroc.compute()
                    test_cm       = test_confmat.compute()
                    test_tn, test_fp, test_fn, test_tp = test_cm.view(-1).tolist()
                    test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
                    test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0                            
                    test_accuracy.reset()
                    test_f1.reset()
                    test_precision.reset()
                    test_recall.reset()
                    test_auroc.reset()
                    test_confmat.reset()
                #---------------------------------------------------------------------------
                
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    # TESTING ------------------------------------------------------------------
                    # model.eval()
                    # with torch.no_grad():
                    #     test_accuracy.reset()
                    #     test_f1.reset()
                    #     test_precision.reset()
                    #     test_recall.reset()
                    #     test_auroc.reset()
                    #     test_confmat.reset()           
                    #     for test_data in test_dataloader:
                    #         if contrastive_mode == "None":
                    #             test_images, test_labels = test_data['data'].to(device), test_data['target'].to(device)
                    #             test_outputs = model(test_images).to(device)
                
                    #         test_preds = test_outputs.argmax(dim=1)
                    #         test_labels = test_labels.argmax(dim=1)
                    #         test_accuracy.update(test_preds, test_labels)
                    #         test_f1.update(test_preds, test_labels)
                    #         test_precision.update(test_preds, test_labels)
                    #         test_recall.update(test_preds, test_labels)
                    #         test_auroc.update(F.softmax(test_outputs, dim=1)[:, 1], test_labels)
                    #         test_confmat.update(test_preds, test_labels)
                    #     torch.distributed.barrier()
                    #     test_acc      = test_accuracy.compute()
                    #     test_f1_score = test_f1.compute()
                    #     test_prec     = test_precision.compute()
                    #     test_rec      = test_recall.compute()
                    #     test_auc      = test_auroc.compute()
                    #     test_cm       = test_confmat.compute()
                    #     test_tn, test_fp, test_fn, test_tp = test_cm.view(-1).tolist()
                    #     test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
                    #     test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0                            
                    #     test_accuracy.reset()
                    #     test_f1.reset()
                    #     test_precision.reset()
                    #     test_recall.reset()
                    #     test_auroc.reset()
                    #     test_confmat.reset()
                    #---------------------------------------------------------------------------
                    if dist.get_rank() == 0:
                        torch.save({
                            'epoch': best_metric_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 
                            'val_auc': val_metric,
                            'train_loss': epoch_loss}, 
                            os.path.join(save_dir, "best_metric_model_classification3d_array.pth"))
                        print("saved new best metric model")
                        
                        print(
                            "test epoch: {} test auc_roc: {:.4f} test acc: {:.4f} test f1: {:.4f} \n test recall: {:.4f} test prec: {:.4f} test spec: {:.4f} \n test tp: {:.4f} test fp: {:.4f} test tn: {:.4f} test fn: {:.4f}".format(
                                epoch + 1, test_auc, test_acc, test_f1_score, test_rec, test_prec, test_specificity, test_tp, test_fp, test_tn, test_fn
                            )
                        )
                    
                if dist.get_rank() == 0:
                    print(
                        "current epoch: {} current auc_roc: {:.4f} current acc: {:.4f} current f1: {:.4f} "
                        "\n current recall: {:.4f} current prec: {:.4f} current spec: {:.4f} "
                        "\n current tp: {} current fp: {} current tn: {} current fn: {} "
                        "\n best auc_roc: {:.4f} at epoch {}".format(
                            epoch + 1, 
                            auc, 
                            acc, 
                            f1_score, 
                            rec,  
                            prec, 
                            specificity,  
                            int(tp), 
                            int(fp),  
                            int(tn),  
                            int(fn), 
                            best_metric, 
                            best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_auc_roc", auc, epoch + 1)
                    writer.add_scalar("val_acc", acc, epoch + 1)
                    writer.add_scalar("val_f1", f1_score, epoch + 1)
                    writer.add_scalar("val_recall", rec,  epoch + 1)
                    writer.add_scalar("val_prec", prec, epoch + 1)
                    writer.add_scalar("val_spec", specificity, epoch + 1)
                    writer.add_scalar("val_sens", sensitivity, epoch + 1)
                    writer.add_scalar("val_tp", tp, epoch + 1)
                    writer.add_scalar("val_fp", fp, epoch + 1)
                    writer.add_scalar("val_tn", tn, epoch + 1)
                    writer.add_scalar("val_fn", fn, epoch + 1)
                    writer.add_scalar("test_auc_roc", test_auc, epoch + 1)
                    writer.add_scalar("test_acc",     test_acc, epoch + 1)
                    writer.add_scalar("test_f1",      test_f1_score, epoch + 1)
                    writer.add_scalar("test_recall",  test_rec,  epoch + 1)
                    writer.add_scalar("test_prec",    test_prec, epoch + 1)
                    writer.add_scalar("test_spec",    test_specificity, epoch + 1)
                    writer.add_scalar("test_sens",    test_sensitivity, epoch + 1)
                    writer.add_scalar("test_tp",      test_tp, epoch + 1)
                    writer.add_scalar("test_fp",      test_fp, epoch + 1)
                    writer.add_scalar("test_tn",      test_tn, epoch + 1)
                    writer.add_scalar("test_fn",      test_fn, epoch + 1)
                
                    
    print(f"train completed, best_auc: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
