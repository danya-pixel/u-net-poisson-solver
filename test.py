import pytorch_lightning as pl
import torch

from dataset import PoissonDataModule


def run_test(config, model_config, run):
    pl.seed_everything(config["SEED"])

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    dm = PoissonDataModule(data_dir=config["DATA_DIR"], batch_size=32, num_workers=16)
    dm.setup()

    artifact = run.use_artifact(model_config["checkpoint"], type="model")
    artifact_dir = artifact.download()

    model = model_config["module"].load_from_checkpoint(artifact_dir + "/model.ckpt")
    trainer = pl.Trainer(
        gpus=1 if str(device) == "cuda:0" else 0, deterministic=False, precision=64
    )

    results = trainer.predict(model, dm.test_dataloader(), return_predictions=True)
    return results
