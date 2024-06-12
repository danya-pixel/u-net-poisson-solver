from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from torch import nn

from train import train_model
from UNet.lightning_modules import NestedUNetModule, UNetModule
from UNet.models import UNetAvg

if __name__ == "__main__":
    print(f"torch: {torch.__version__}, pytorch lightning: {pl.__version__}")

    SEED = 42
    pl.seed_everything(SEED)

    wandb.login()

    BATCH_SIZE = 64
    SHAPE = 128
    N_SAMPLES = 5000

    data_dir = Path(f"data_s{SHAPE}_n{N_SAMPLES}")

    lightning_modules = [UNetModule]  # you can add modules
    models = [UNetAvg]  # you can add models
    bilinear_type = [True]  # hyperparameters: bilinear or transpose conv

    for module in lightning_modules:
        for bilinear in bilinear_type:
            unet_model, test_preds = train_model(
                base_module=module,
                model_hparams={
                    "epochs": 50,
                    "batch_size": BATCH_SIZE,
                    "SHAPE": SHAPE,
                    "N_SAMPLES": N_SAMPLES,
                    "filters": [4, 8, 16, 32, 64, 128, 256],
                    "bilinear": bilinear,
                    "in_channels": 1,
                    "out_channels": 1,
                },
                data_dir=data_dir,
                optimizer_name="Adam",
                optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
                samples_num=N_SAMPLES,
                loss_module=nn.MSELoss(),
                project="PoissonNN128",
            )
            wandb.finish()
