from typing import Dict

import pytorch_lightning as pl
import torch.optim as optim
from torch import nn, norm

from .models import *


class BaseUNetModule(pl.LightningModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()

        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.save_hyperparameters()

        self.loss_module = nn.MSELoss()
        self.model = None

    def forward(self, x):
        pred = self.model(x)
        return pred

    def predict_step(self, batch, batch_idx):
        x, y, bound = batch
        pred = self.model(x)

        return pred, y, x, bound

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        pred = self.model(x)
        loss = self.loss_module(pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        pred = self.model(x)
        loss = self.loss_module(pred, y)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        pred = self.model(x)
        loss = self.loss_module(pred, y)
        relative_loss = norm(pred - y) / norm(y)

        self.log("test_loss", loss)
        self.log("relative_loss", relative_loss)

    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20, 30], gamma=0.5
        )
        return [optimizer], [scheduler]


class UNetModule(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)

        self.model = UNet(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )


class UNet512Module(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)

        self.model = UNet512(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )


class UNet512mModule(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)

        self.model = UNet512m(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )


class UNetAvg1024Module300k(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)

        self.model = UNet1024Avg300k(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )


class UNetAvg2048Module300k(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)
        self.model = UNet2048Avg300k(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )


class UNetAvg2048Module850k(BaseUNetModule):
    def __init__(
        self, model_hparams: Dict, optimizer_name: str, optimizer_hparams: Dict
    ):
        """Initialize an UNet model

        Args:
            model_hparams (Dict): keys: batch_size: int, num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__(model_hparams, optimizer_name, optimizer_hparams)

        self.model = UNet2048Avg850k(
            in_channels=1,
            out_channels=1,
            is_bilinear=model_hparams.get("is_bilinear"),
            is_avg=model_hparams.get("is_avg"),
            is_leaky=model_hparams.get("is_leaky"),
        )
