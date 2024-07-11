import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from dataset import PoissonDataModule
from UNet.lightning_modules import UNetModule, UNetAvgModule


def train_model(
    base_module,
    model_hparams,
    data_dir,
    project,
    **kwargs,
):
    """
    Inputs:
        model_hparams: Dict - model_name, epochs, num_classes, act_fn_name, batch_size
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    model = base_module(model_hparams=model_hparams, **kwargs)
    # ---initialize Wandb logger for pytorch lightning---

    if model_hparams.get("bilinear") == True:
        model_name = model.model.__class__.__name__ + "_Interpolation"
    else:
        model_name = model.model.__class__.__name__ + "_Transposed"

    wandb_logger = WandbLogger(
        project=project,
        name=model_name,
        log_model="all",
        reinit=True,
    )
    # ---initialize DataModule---#
    CHECKPOINT_PATH = Path("pretrained_models")
    os.makedirs(CHECKPOINT_PATH / model_name, exist_ok=True)

    dm = PoissonDataModule(
        data_dir=data_dir,
        batch_size=model_hparams.get("batch_size"),
        num_workers=16,
    )
    # ---initialize Trainer---#
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
        devices=1,
        max_epochs=model_hparams["epochs"],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=142),
        ],
        logger=wandb_logger,
        deterministic=not model_hparams.get(
            "bilinear"
        ),  # upsample.backwards requires deterministic=False
        precision=64,
        accelerator="gpu",
    )

    wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=dm)
    model = base_module.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # ---test best model on validation and test set---#
    trainer.validate(model, datamodule=dm, verbose=True)
    trainer.test(model, datamodule=dm, verbose=True)
    test_preds = trainer.predict(model, datamodule=dm, return_predictions=True)

    return model, test_preds


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
                    "filters": [4, 8, 16, 32, 64],
                    "bilinear": bilinear,
                    "in_channels": 1,
                    "out_channels": 1,
                },
                data_dir=data_dir,
                optimizer_name="Adam",
                optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
                project="PoissonU-Net",
            )
            wandb.finish()
