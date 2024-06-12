import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger

from dataset import PoissonDataModule


def train_model(
    base_module,
    model_hparams,
    data_dir,
    samples_num,
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
        model_name = base_module.model.__name__ + "B"
    else:
        model_name = base_module.model.__name__

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
        num_workers=24,
        samples_num=samples_num,
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


def run_test(config, model_config, run):
    pl.seed_everything(config["SEED"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    dm = PoissonDataModule(
        data_dir=config["DATA_DIR"], batch_size=config["BATCH_SIZE"], num_workers=16
    )

    dm.setup()
    print("pth" in model_config.keys())
    if "pth" in model_config.keys():
        model = model_config["module"].load_from_checkpoint(model_config["pth"])
    else:
        artifact = run.use_artifact(model_config["checkpoint"], type="model")
        artifact_dir = artifact.download()

        model = model_config["module"].load_from_checkpoint(
            artifact_dir + "/model.ckpt"
        )

    trainer = pl.Trainer(
        num_sanity_val_steps=2,
        devices=1,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
        ],
        precision=64,
        accelerator="gpu",
    )
    print("predicting")
    print(len(dm.test_dataloader()))
    results = trainer.predict(model, dm.test_dataloader(), return_predictions=True)
    print("finished calcs")
    return results
