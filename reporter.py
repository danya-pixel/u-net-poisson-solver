from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import TQDMProgressBar

from dataset import PoissonDataModule
from utils import get_models_configs, get_relative_error_stats
from vizualize import *


class Reporter:
    def __init__(self, config) -> None:
        self.config = config
        self.config["DATA_DIR"] = f"data_s{config['SHAPE']}_n{config['N_SAMPLES']}"

        self.models = get_models_configs()[config["SHAPE"]]
        self.run = wandb.init()
        print(torch.cuda.is_available())

        self.all_modules_errors = {}
        self.all_modules_cuts = {}

    def dump_all_models_stats(self):
        # errors boxplot
        boxplot = get_boxplot(self.all_modules_errors)
        boxplot.write_image(
            Path(self.config["SAVE_DIR"]) / "box.png",
            scale=self.config["PLOT_SCALE_FACTOR"],
        )

        # cuts
        for example in self.all_modules_cuts:
            SAVE_DIR = Path(f"{self.config['SAVE_DIR']}") / f"{example}"
            SAVE_DIR.mkdir(exist_ok=True, parents=True)
            example_cuts = self.all_modules_cuts[example]
            for cut_type in example_cuts:
                all_modules_stats = []
                for module in self.all_modules_cuts[example][cut_type]:
                    all_modules_stats.append(
                        (self.all_modules_cuts[example][cut_type][module], module)
                    )

                fig = get_full_plot2D(all_modules_stats, self.config["SHAPE"], cut_type)
                fig.write_image(
                    SAVE_DIR / f"_{cut_type}.png",
                    scale=self.config["PLOT_SCALE_FACTOR"],
                )

    def test_module(self, bilinear_type: str, module: str):
        print(f"testing {module}")
        results = run_test(
            config=self.config,
            model_config=self.models[bilinear_type][module],
            run=self.run,
            # run=None,
        )
        pred_flat, true_flat, errors = get_relative_error_stats(
            results=results, save_preds=self.config["SAVE_PREDS"]
        )
        # make graphs from one model
        self.write_single_model_graphs(module, bilinear_type, pred_flat, true_flat)
        # make and save cuts
        self.save_model_cuts(module, bilinear_type, pred_flat, true_flat)
        # save errors stats
        self.save_module_errors(module, bilinear_type, errors)

        print(f"Mean relative error: {np.mean(errors)}")

    def save_model_cuts(self, module, bilinear_type, pred_flat, true_flat):
        for idx in self.config["PLOT_INDEXES"]:
            self.init_model_cuts_dict(idx, true_flat[idx])

            x_border, y_border, x_mid, y_mid = self.get_cuts(pred_flat[idx])
            model_key = module + f"_{bilinear_type}"
            self.all_modules_cuts[idx]["x_border"][model_key] = x_border
            self.all_modules_cuts[idx]["y_border"][model_key] = y_border
            self.all_modules_cuts[idx]["x_mid"][model_key] = x_mid
            self.all_modules_cuts[idx]["y_mid"][model_key] = y_mid

    def init_model_cuts_dict(self, idx, true):
        if idx not in self.all_modules_cuts:
            self.all_modules_cuts[idx] = defaultdict(lambda: defaultdict(int))
            x_border, y_border, x_mid, y_mid = self.get_cuts(true)
            self.all_modules_cuts[idx]["x_border"]["true"] = x_border
            self.all_modules_cuts[idx]["y_border"]["true"] = y_border
            self.all_modules_cuts[idx]["x_mid"]["true"] = x_mid
            self.all_modules_cuts[idx]["y_mid"]["true"] = y_mid

    def get_cuts(self, tensor):
        x_border = tensor[0, :]
        y_border = tensor[:, 0]
        x_mid = tensor[self.config["SHAPE"] // 2, :]
        y_mid = tensor[:, self.config["SHAPE"] // 2]

        return x_border, y_border, x_mid, y_mid

    def write_single_model_graphs(
        self, module: str, bilinear_type, pred_flat, true_flat
    ):
        for idx in self.config["PLOT_INDEXES"]:
            SAVE_DIR = Path(
                f"{self.config['SAVE_DIR']}/{module+f'_{bilinear_type}'}/{idx}"
            )
            SAVE_DIR.mkdir(exist_ok=True, parents=True)
            # 3D pics
            pred3D = get_plot3D(surface=pred_flat[idx], title="<i>pred</i>")
            true3D = get_plot3D(surface=true_flat[idx], title="<i>true</i>")

            pred3D.write_image(
                SAVE_DIR / f"pred3D{idx}.png", scale=self.config["PLOT_SCALE_FACTOR"]
            )
            true3D.write_image(
                SAVE_DIR / f"true3D{idx}.png", scale=self.config["PLOT_SCALE_FACTOR"]
            )

            # errors heatmap
            heatmap_percent = get_error_heatmap(
                pred=pred_flat[idx], true=true_flat[idx]
            )
            heatmap_percent.write_image(
                SAVE_DIR / f"heatmap_percent{idx}.png",
                scale=self.config["PLOT_SCALE_FACTOR"],
            )

    def save_module_errors(self, module, bilinear_type, errors):
        self.all_modules_errors[module + f"_{bilinear_type}"] = errors


def run_test(config, model_config, run):
    pl.seed_everything(config["SEED"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    dm = PoissonDataModule(
        data_dir=config["DATA_DIR"], batch_size=config["BATCH_SIZE"], num_workers=16
    )

    dm.setup()

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
    print(f"num of samples: {len(dm.test_dataloader())}")
    results = trainer.predict(model, dm.test_dataloader(), return_predictions=True)
    print("finished calcs")
    return results
