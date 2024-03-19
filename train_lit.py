import os
import torch
import torch.nn.functional as F
import numpy as np
from monai.transforms import (
    Compose,
    OneOf,  
    RandGaussianNoise,
    RandScaleIntensity,
    RandAdjustContrast,
    RandAffine,
    Identity
)
from torch.utils.data import DataLoader
from classification.dataloader import ScanDataset, MRNDataset, MacOpDataset
from classification.model_factory import model_factory, FocalLoss
from classification.dataloader_utils import split_and_process
import pandas as pd

from torchmetrics.classification import Accuracy, F1Score, AUROC, Specificity, Recall, ConfusionMatrix
import wandb
# from torchmetrics.functional.classification import binary_specificity_at_sensitivity
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn



import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
import finetuning_scheduler as fts


class CustomLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        model_config = self.config.fit.model
        data_config = self.config.fit.data
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        image_size = '_'.join(map(str, model_config.image_size))
        custom_name = f"{model_config.model_name}d{model_config.dropout}b{data_config.batch_size}lr{model_config.lr}img{image_size}_{current_time}{model_config.contrastive_mode}{data_config.dataset_name}"
        self.config.fit.trainer.logger.init_args.name = custom_name
        super().before_instantiate_classes()

class GlaucomaModel(L.LightningModule):
    def __init__(
            self, 
            model_name="ResNext50", 
            image_size=(128,200,200), 
            dropout=.4, 
            num_classes=1, 
            contrastive_mode="None", 
            conv_layers=[32,64], 
            fc_layers=[32], 
            freeze=False, 
            lr=5e-5, 
            weight_decay=1e-2, 
            pos_weight=1, 
            use_dual_paths=False, 
            cos_anneal=-1, 
            step_decay=-1):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory(model_name, image_size, dropout=dropout, num_classes=num_classes, contrastive_mode=contrastive_mode, conv_layers=conv_layers, fc_layers=fc_layers, freeze=freeze, use_dual_paths=use_dual_paths)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.train_acc = Accuracy(task="binary")
        self.acc = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary")
        self.spec = Specificity(task="binary")        
        self.sens = Recall(task="binary")
        self.conf = ConfusionMatrix(task="binary")



    def forward(self, x, aux=None):
        if self.hparams.contrastive_mode=="None":
            return self.model(x)
        else:
            return self.model(x, aux)

    def training_step(self, batch, batch_idx):
        if self.hparams.contrastive_mode == "None":
            x, y = batch['data'], batch['target']
            logits = self(x)
        else:
            x, aux, y = batch['data'], batch['aux'], batch['target']
            logits = self(x, aux)
        
        loss = self.loss_fn(logits, y)
        # acc = accuracy((logits > 0.5), y, task='binary')
        self.train_acc((logits > 0.5), y)

        self.log('train_loss', loss)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        return loss
    
    def evaluate(self, batch, stage=None):
        if self.hparams.contrastive_mode == "None":
            x, y = batch['data'], batch['target']
            logits = self(x)
        else:
            x, aux, y = batch['data'], batch['aux'], batch['target']
            logits = self(x, aux)
        preds = (logits > 0.5)

        self.acc(preds, y)
        self.f1(preds, y)
        self.spec(preds, y)
        self.sens(preds, y)
        self.auroc(logits, y)
        self.conf.update(preds, y)
        loss = self.loss_fn(logits, y)

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
            self.log(f"{stage}_acc", self.acc,   on_epoch=True, on_step=False)
            self.log(f"{stage}_f1", self.f1,     on_epoch=True, on_step=False)
            self.log(f"{stage}_sens", self.sens, on_epoch=True, on_step=False)
            self.log(f"{stage}_spec", self.spec, on_epoch=True, on_step=False)
            self.log(f"{stage}_auc", self.auroc, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def on_validation_epoch_end(self):
        cm = self.conf.compute()
        df_cm = pd.DataFrame(cm.cpu().numpy() , index = [0,1], columns = [0,1])
        f, ax = plt.subplots(figsize = (20,15)) 
        sn.heatmap(df_cm, annot=True, ax=ax)
        wandb.log({"plot": wandb.Image(f) })
        self.conf.reset() 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.cos_anneal > 0:
            scheduler = CosineAnnealingLR(optimizer, T_max = self.hparams.cos_anneal)
            return [optimizer], [scheduler]
        elif self.hparams.step_decay > 0:
            scheduler = StepLR(optimizer, self.hparams.step_decay)
            return [optimizer], [scheduler]
        return optimizer
    
class OCTDataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset_name="Optic14.csv", 
        batch_size=8, 
        num_workers=4, 
        image_size=(128,200,200), 
        split_name="split1", 
        tv=True, 
        contrastive_mode="None", 
        prob=.4, 
        add_denoise=True, 
        num_classes=1, 
        mrn_mode=False, 
        imbalance_factor=-1
    ):
        super().__init__()
        self.save_hyperparameters()     
        self.training_data = ["train"] if not tv else ["train", "val"]

    def setup(self, stage=None):
        transforms_list = Compose([
        RandGaussianNoise(prob=self.hparams.prob), 
        RandScaleIntensity(prob=self.hparams.prob, factors=(1,4)),
        RandAdjustContrast(prob=self.hparams.prob),
        RandAffine(prob=self.hparams.prob, translate_range=(15,10, 0), rotate_range=(0.02,0,0), scale_range=((-.1, .4), 0,0), padding_mode = "zeros"),
        ])
        transforms = OneOf([transforms_list, Identity()], weights=[self.hparams.prob, 1 - self.hparams.prob])

        if stage == "fit":
            if self.hparams.dataset_name == "Macop":
                self.train_dataset = MacOpDataset("Macular13.csv", "Optic13.csv", self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise)
                self.val_dataset = MacOpDataset("Macular13.csv", "Optic13.csv", self.hparams.split_name, ["test"], transforms, self.hparams.image_size, self.hparams.add_denoise)
            else:
                if self.hparams.mrn_mode:
                    self.train_dataset = MRNDataset(self.hparams.dataset_name, self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode)
                else:
                    self.train_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)
                self.val_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, ["test"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)
            
        
        if stage == "test":
            if self.hparams.dataset_name == "Macop":
                self.test_dataset = MacOpDataset("Macular13.csv", "Optic13.csv", self.hparams.split_name, ["val"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode)
            else:
                if self.hparams.mrn_mode:
                    self.test_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, ["val"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode)
                else:
                    self.test_dataset = MRNDataset(self.hparams.dataset_name, self.hparams.split_name, ["val"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode)
        print("Finished processing data") 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

def cli_main():
    cli = CustomLightningCLI(GlaucomaModel, OCTDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()
    


