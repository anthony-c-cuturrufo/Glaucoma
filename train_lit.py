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
        torch.set_float32_matmul_precision('medium')
        model_config = self.config.fit.model
        data_config = self.config.fit.data
        trainer_config = self.config.fit.trainer
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        image_size = '_'.join(map(str, model_config.image_size))
        custom_name = f"{model_config.model_name}d{model_config.dropout}b{data_config.batch_size}lr{model_config.lr}img{image_size}_{current_time}{model_config.contrastive_mode}{data_config.dataset_name}b_acc{trainer_config.accumulate_grad_batches}"
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
            patch_s=18, 
            hidden_s=768,
            mlp_d=3072,
            num_l=12,
            num_h=12,
            qkv=False,
            freeze=False, 
            lr=5e-5, 
            weight_decay=1e-2, 
            pos_weight=1, 
            use_dual_paths=False, 
            cos_anneal=-1, 
            step_decay=-1):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory(
                        model_name, 
                        image_size, 
                        dropout=dropout, 
                        num_classes=num_classes, 
                        contrastive_mode=contrastive_mode, 
                        conv_layers=conv_layers, 
                        fc_layers=fc_layers, 
                        freeze=freeze, 
                        use_dual_paths=use_dual_paths,
                        patch_s=patch_s, 
                        hidden_s=hidden_s,
                        mlp_d=mlp_d,
                        num_l=num_l,
                        num_h=num_h,
                        qkv=qkv        
                    )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_spec = Specificity(task="binary")        
        self.val_sens = Recall(task="binary")
        self.val_conf = ConfusionMatrix(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_spec = Specificity(task="binary")        
        self.test_sens = Recall(task="binary")
        self.test_conf = ConfusionMatrix(task="binary")

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
        self.train_acc((logits > 0.5), y)

        self.log('train_loss', loss, sync_dist=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.contrastive_mode == "None":
            x, y = batch['data'], batch['target']
            logits = self(x)
        else:
            x, aux, y = batch['data'], batch['aux'], batch['target']
            logits = self(x, aux)
        preds = (logits > 0.5)
        loss = self.loss_fn(logits, y)

        if dataloader_idx == 0:
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            self.val_spec(preds, y)
            self.val_sens(preds, y)
            self.val_auroc(logits, y)
            self.val_conf.update(preds, y)
            self.log("val_loss", loss, on_epoch=True, on_step=False,add_dataloader_idx=False)
            self.log("val_acc", self.val_acc,  on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("val_f1",  self.val_f1,   on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("val_sens",self.val_sens, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("val_spec",self.val_spec, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("val_auc", self.val_auroc,on_epoch=True, on_step=False, add_dataloader_idx=False)
        else:
            self.test_acc(preds, y)
            self.test_f1(preds, y)
            self.test_spec(preds, y)
            self.test_sens(preds, y)
            self.test_auroc(logits, y)
            self.test_conf.update(preds, y)
            self.log("test_loss", loss, on_epoch=True, on_step=False,add_dataloader_idx=False)
            self.log("test_acc", self.test_acc,  on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("test_f1",  self.test_f1,   on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("test_sens",self.test_sens, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("test_spec",self.test_spec, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log("test_auc", self.test_auroc,on_epoch=True, on_step=False, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx):
        if self.hparams.contrastive_mode == "None":
            x, y = batch['data'], batch['target']
            logits = self(x)
        else:
            x, aux, y = batch['data'], batch['aux'], batch['target']
            logits = self(x, aux)
        preds = (logits > 0.5)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_spec(preds, y)
        self.test_sens(preds, y)
        self.test_auroc(logits, y)
        self.test_conf.update(preds, y)
        loss = self.loss_fn(logits, y)

        stage = "test"
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", self.test_acc,   on_epoch=True, on_step=False)
        self.log(f"{stage}_f1",  self.test_f1,     on_epoch=True, on_step=False)
        self.log(f"{stage}_sens",self.test_sens, on_epoch=True, on_step=False)
        self.log(f"{stage}_spec",self.test_spec, on_epoch=True, on_step=False)
        self.log(f"{stage}_auc", self.test_auroc, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        cm = self.val_conf.compute()
        df_cm = pd.DataFrame(cm.cpu().numpy(), index = [0,1], columns = [0,1])
        f, ax = plt.subplots(figsize = (20,15)) 
        sn.heatmap(df_cm, annot=True, ax=ax)
        wandb.log({"plot": wandb.Image(f) })
        self.val_conf.reset() 
        plt.close(f)

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
        dataset_name="Optic15.csv", 
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
                self.train_dataset = MacOpDataset("Macular15.csv", "Optic15.csv", self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise)
                self.val_dataset = MacOpDataset("Macular15.csv", "Optic15.csv", self.hparams.split_name, ["test"], transforms, self.hparams.image_size, self.hparams.add_denoise)
            else:
                if self.hparams.mrn_mode:
                    self.train_dataset = MRNDataset(self.hparams.dataset_name, self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)
                else:
                    self.train_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, self.training_data, transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)
                self.val_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, ["test"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)
            
        
        if stage == "test" or not self.hparams.tv:
            if self.hparams.dataset_name == "Macop":
                self.test_dataset = MacOpDataset("Macular15.csv", "Optic15.csv", self.hparams.split_name, ["val"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode)
            else:
                self.test_dataset = ScanDataset(self.hparams.dataset_name, self.hparams.split_name, ["val"], transforms, self.hparams.image_size, self.hparams.add_denoise, self.hparams.contrastive_mode, self.hparams.imbalance_factor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        if not self.hparams.tv:
            return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False), 
                DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)]
        else:
            return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

def cli_main():
    cli = CustomLightningCLI(GlaucomaModel, OCTDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()
    


