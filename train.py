import os
import torch
import torch.nn as nn
from monai.transforms import (
    Compose, OneOf, RandGaussianNoise, RandScaleIntensity, 
    RandAdjustContrast, RandAffine, Identity, RandFlip, NormalizeIntensity
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from classification.dataloader import ScanDataset, MRNDataset, MacOpDataset, HiroshiScan
from classification.model_factory import model_factory
from torchmetrics.classification import (
    Accuracy, F1Score, AUROC, Specificity, Recall, ConfusionMatrix, AveragePrecision
)
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from datetime import datetime
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningCLI


class CustomLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        torch.set_float32_matmul_precision('medium')
        print(self.config)
        model_config, data_config, trainer_config = self.config.fit.model, self.config.fit.data, self.config.fit.trainer
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        image_size = '_'.join(map(str, model_config.image_size))
        add_D = "_add" if data_config.add_denoise else ""
        post_norm = "_pn" if data_config.post_norm else ""
        dall = "_dall" if data_config.denoise_all else ""
        ws = "_ws" if data_config.weighted_sampling else ""
        custom_name = (f"{model_config.model_name}d{model_config.dropout}b{data_config.batch_size}{add_D}{post_norm}{dall}{ws}"
                       f"lr{model_config.lr}img{image_size}_{current_time}{model_config.contrastive_mode}"
                       f"{data_config.dataset_name}b_acc{trainer_config.accumulate_grad_batches}{data_config.split_name}")
        self.config.fit.trainer.logger.init_args.name = custom_name
        self.config.fit.trainer.default_root_dir = os.path.join("/local2/acc/Glaucoma/logs", custom_name)
        self.config.fit.trainer.callbacks[0].init_args.dirpath = self.config.fit.trainer.default_root_dir
        super().before_instantiate_classes()

    def after_fit(self):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            self.trainer.test(ckpt_path=best_ckpt_path, datamodule=self.datamodule)
        else:
            print("No best checkpoint available, skipping testing.")


class GlaucomaModel(L.LightningModule):
    def __init__(self, lr=1e-5, weight_decay=1e-2, cos_anneal=-1, step_decay=-1, pos_weight=1, eps=1e-8, opt="Adam", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hparams.pos_weight]))
        metric_classes = [
            Accuracy, F1Score, AUROC, Specificity, Recall, AveragePrecision
        ]
        self.train_metrics = nn.ModuleList(self._init_metrics(metric_classes))
        self.val_metrics = nn.ModuleList(self._init_metrics(metric_classes))
        self.test_metrics = nn.ModuleList(self._init_metrics(metric_classes))

    def _init_metrics(self, metric_classes):
        return [cls(task="binary") for cls in metric_classes]

    def forward(self, x, aux=None):
        return self.model(x) if self.hparams.contrastive_mode == "None" else self.model(x, aux)

    def _shared_step(self, batch, stage):
        x, y = batch['data'], batch['target']
        aux = batch.get('aux')
        logits = self(x) if aux is None else self(x, aux)
        preds = (logits > 0.5).float()
        y = y.long()
        loss = self.loss_fn(logits, y.float())
        
        metrics = getattr(self, f"{stage}_metrics")
        metric_names = ["acc", "f1", "auroc", "spec", "sens", "auprc"]
        
        for metric, name in zip(metrics, metric_names):
            if name in ["auroc", "auprc"]:
                metric(logits, y)
            else:
                metric(preds, y)
            self.log(f"{stage}_{name}", metric, on_epoch=True, sync_dist=True, logger=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        stage = "val" if dataloader_idx == 0 else "test"
        loss = self._shared_step(batch, stage)
        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx=1)

    def configure_optimizers(self):
        optimizer_class = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "NAdam": torch.optim.NAdam
        }.get(self.hparams.opt, torch.optim.Adam)

        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps
        )

        if self.hparams.cos_anneal > 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.cos_anneal)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif self.hparams.step_decay > 0:
            scheduler = StepLR(optimizer, step_size=self.hparams.step_decay)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
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
        imbalance_factor=-1,
        post_norm=False,
        denoise_all=True,
        denoise_prob=.4, 
        transform_denoise=True,
        weighted_sampling=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.training_data = ["train"] if not tv else ["train", "val"]

    def setup(self, stage=None):
        prob, dataset_name, split_name = self.hparams.prob, self.hparams.dataset_name, self.hparams.split_name
        if self.hparams.post_norm:
            train_transform = Compose([
                    # RandGaussianNoise(prob=prob, std=0.01 if True else 1),
                    # RandScaleIntensity(prob=prob, factors=(0.9, 1.1) if True else (1, 4)),
                    RandAdjustContrast(prob=prob, gamma=(0.9, 1.1) if True else 1),
                    RandAffine(
                        prob=prob, translate_range=(5, 5, 0) if True else (15, 10, 0),
                        rotate_range=(0.01, 0, 0) if True else (0.02, 0, 0),
                        scale_range=((-.1, .1), 0, 0) if True else ((-.1, .4), 0, 0),
                        padding_mode="zeros"
                    ),
                    RandFlip(spatial_axis=0, prob=prob),
                    RandFlip(spatial_axis=1, prob=prob),
                    NormalizeIntensity(nonzero=True)
                ]).set_random_state(seed=1)
            val_transform = Compose([NormalizeIntensity(nonzero=True)])
            test_transform = Compose([NormalizeIntensity(nonzero=True)])   
        else:
            train_transform = Compose([
                # RandGaussianNoise(prob=prob, std=0.01 if True else 1),
                # RandScaleIntensity(prob=prob, factors=(0.9, 1.1) if True else (1, 4)),
                RandAdjustContrast(prob=prob, gamma=(0.9, 1.1) if True else 1),
                RandAffine(
                    prob=prob, translate_range=(5, 5, 0) if True else (15, 10, 0),
                    rotate_range=(0.01, 0, 0) if True else (0.02, 0, 0),
                    scale_range=((-.1, .1), 0, 0) if True else ((-.1, .4), 0, 0),
                    padding_mode="zeros"
                ),
                RandFlip(spatial_axis=0, prob=prob),
                RandFlip(spatial_axis=1, prob=prob)
            ]).set_random_state(seed=1)
        
            val_transform = None
            test_transform = None 

        DatasetClass = HiroshiScan

        self.train_dataset = DatasetClass(dataset_name, split_name, self.training_data, train_transform, 
                                          self.hparams.image_size, self.hparams.add_denoise,
                                          self.hparams.contrastive_mode, self.hparams.imbalance_factor,
                                          self.hparams.post_norm, self.hparams.denoise_all,
                                          self.hparams.denoise_prob, self.hparams.transform_denoise)
        self.val_dataset = DatasetClass(dataset_name, split_name, ["val"], val_transform,
                                        self.hparams.image_size, self.hparams.add_denoise,
                                        self.hparams.contrastive_mode, self.hparams.imbalance_factor,
                                        self.hparams.post_norm, self.hparams.denoise_all,
                                        self.hparams.denoise_prob, self.hparams.transform_denoise)
        self.test_dataset = DatasetClass(dataset_name, split_name, ["test"], test_transform,
                                            self.hparams.image_size, self.hparams.add_denoise,
                                            self.hparams.contrastive_mode, self.hparams.imbalance_factor,
                                            self.hparams.post_norm, self.hparams.denoise_all,
                                            self.hparams.denoise_prob, self.hparams.transform_denoise)

    def train_dataloader(self):
        if self.hparams.weighted_sampling:
            targets = [self.train_dataset[i]['target'] for i in range(len(self.train_dataset))]
            targets = torch.tensor(targets).long()
            class_counts = torch.bincount(targets)
            class_weights = 1.0 / class_counts.float()
            sample_weights = [class_weights[target] for target in targets]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, sampler=sampler, num_workers=self.hparams.num_workers)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)


def cli_main():
    seed_everything(1, workers=True)
    cli = CustomLightningCLI(GlaucomaModel, OCTDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()