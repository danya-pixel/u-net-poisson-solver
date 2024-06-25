from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytorch_lightning as pl
import torch.optim as optim
import wandb
from torch import nn, norm
from torch.nn.functional import interpolate

from vizualize import get_plot2D, get_plot3D

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
        print(model_hparams.get("bilinear"))
        self.model = UNet(
            in_channels=1, out_channels=1, bilinear=model_hparams.get("bilinear")
        )


class UNetAvgModule(BaseUNetModule):
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

        self.model = UNetAvg(
            in_channels=1, out_channels=1, bilinear=model_hparams.get("bilinear")
        )


class UNetLeakyModule(BaseUNetModule):
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

        self.model = UNetLeaky(
            in_channels=1, out_channels=1, bilinear=model_hparams.get("bilinear")
        )


class UNetLeakyAvgModule(BaseUNetModule):
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

        self.model = UNetLeakyAvg(
            in_channels=1, out_channels=1, bilinear=model_hparams.get("bilinear")
        )


class NestedUNetModule(BaseUNetModule):
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

        self.model = NestedUNet(
            in_channels=1,
            out_channels=1,
            # filters=[2, 4, 8, 16, 32, 64],
            filters=[4, 8, 8, 32, 64, 128],
        )


class NestedUNetModuleT(BaseUNetModule):
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

        self.model = NestedUNetTransposed(
            in_channels=1,
            out_channels=1,
            # filters=[2, 4, 8, 16, 32, 64],
            filters=[4, 8, 8, 32, 64, 128],
        )


class NestedUNetModuleAvg(BaseUNetModule):
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

        self.model = NestedUNetAvg(
            in_channels=1,
            out_channels=1,
            # filters=[2, 4, 8, 16, 32, 64],
            filters=[4, 8, 8, 32, 64, 128],
        )


class NestedUNetModuleLeaky(BaseUNetModule):
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

        self.model = NestedUNetLeaky(
            in_channels=1,
            out_channels=1,
            # filters=[2, 4, 8, 16, 32, 64],
            filters=[4, 8, 8, 32, 64, 128],
        )


class NestedUNetModuleLeakyAvg(BaseUNetModule):
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

        self.model = NestedUNetLeakyAvg(
            in_channels=1,
            out_channels=1,
            # filters=[2, 4, 8, 16, 32, 64],
            filters=[4, 8, 8, 32, 64, 128],
        )


class NestedUNet512Module(BaseUNetModule):
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

        self.model = NestedUNet512(
            in_channels=1,
            out_channels=1,
            filters=[4, 8, 16, 32, 64, 128, 256],
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

        self.model = UNet512(in_channels=1, out_channels=1, bilinear=True)


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

        self.model = UNet512m(in_channels=1, out_channels=1, bilinear=True)


class UNetAvg512Module(BaseUNetModule):
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

        self.model = UNetAvg512(in_channels=1, out_channels=1, bilinear=True)


class NestedUNet512ModuleAvg(BaseUNetModule):
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

        self.model = NestedUNet512Avg(
            in_channels=1,
            out_channels=1,
            filters=[4, 8, 16, 32, 64, 128, 256],
        )


class UNetAvg1024Module(BaseUNetModule):
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

        self.model = UNet1024Avg(in_channels=1, out_channels=1, bilinear=True)


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

        self.model = UNet1024Avg300k(in_channels=1, out_channels=1, bilinear=True)


class UNetAvg1024Module300kResize(BaseUNetModule):
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
        self.model = UNet1024Avg300k(in_channels=1, out_channels=1, bilinear=True)

    def validation_step(self, batch, batch_idx):
        _, resized_x, y = batch
        pred = self.model(resized_x)
        resized_pred = interpolate(pred, y.size()[2:], mode="bicubic")
        loss = self.loss_module(resized_pred, y)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # resized_x = interpolate(x, (1024, 1024), mode='bicubic')
        pred = self.model(x)
        # resized_pred = interpolate(pred, y.size()[2:], mode='bicubic')

        # loss = self.loss_module(resized_pred, y)
        # relative_loss = norm(resized_pred - y) / norm(y)
        loss = self.loss_module(pred, y)
        relative_loss = norm(pred - y) / norm(y)

        self.log("test_loss", loss)
        self.log("relative_loss", relative_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        resized_x = interpolate(x, (1024, 1024), mode="bicubic")
        pred = self.model(resized_x)
        resized_pred = interpolate(pred, y.size()[2:], mode="bicubic")

        if batch_idx == 0:
            size = int(y.size()[2])
            cut = 20
            N = 11
            cutted_pred = resized_pred[N][0][cut : size - cut, cut : size - cut]
            # cutted_resized_pred = pred[N][0][cut:1024-cut, cut:1024-cut]
            cutted_true = y[N][0][cut : size - cut, cut : size - cut]
            DIR = Path(f"plots/{size}_2")
            DIR.mkdir(parents=True, exist_ok=True)
            plot = get_plot3D(resized_pred[N][0].cpu().numpy(), " ")
            # wandb.log({"3d_pred":plot}, step=int(y.size()[2]))
            # plot2d = get_plot2D(resized_pred[8][0][2].cpu().numpy(), y[8][0][2].cpu().numpy(), shape=init_shape)
            plot.write_image(f"{DIR}/pred_3d.png")

            plot = get_plot3D(y[N][0].cpu().numpy(), " ")
            plot.write_image(f"{DIR}/y_3d.png")

            plot = get_plot3D(cutted_pred.cpu().numpy(), " ")
            plot.write_image(f"{DIR}/pred_cut.png")
            # plot2d = get_plot2D(resized_pred[8][0][150].cpu().numpy(), y[8][0][150].cpu().numpy(), shape=init_shape)
            # plot2d.write_image('pred_2.png')

            plot = get_plot3D(cutted_true.cpu().numpy(), " ")
            plot.write_image(f"{DIR}/true_cut.png")

            plot = get_plot3D(pred[N][0].cpu().numpy(), " ")
            plot.write_image(f"{DIR}/unresized_pred.png")
            # wandb.log({"3d_true":plot}, step=int(y.size()[2]))

            # plot_y = get_heatmap(pred=y[2][0].cpu())
            # wandb.log({"heatmap_y":plot_y}, step=int(y.size()[2]))
        return resized_pred, y
        # return cutted_pred, cutted_true


class NestedUNet1024ModuleAvg(BaseUNetModule):
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

        self.model = NestedUNet1024Avg(
            in_channels=1,
            out_channels=1,
            filters=[2, 4, 8, 16, 32, 64, 128, 256],
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
        self.model = UNet2048Avg300k(in_channels=1, out_channels=1, bilinear=True)


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

        self.model = UNet2048Avg850k(in_channels=1, out_channels=1, bilinear=True)
