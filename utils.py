import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import norm

from UNet.lightning_modules import *


def get_models_configs():
    models = {
        128: {
            "bilinear": {
                "UNet": {
                    "module": UNetModule,
                    "checkpoint": "",  # here you can paste wandb checkpoint
                    "pth": "" # or .pth model weights 
                },
            },
        },
    }
    return models


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{kw['info']}, {(te - ts)*1000:.3f} ms")
        return result, (te - ts) * 1000

    return timed


def plot_stats(stats):
    legend, data = zip(*stats.items())
    plt.plot(data[0], data[1])
    plt.xlabel(legend[0])
    plt.ylabel(legend[1])
    plt.show()


def count_receptive_field(N, kernel_size=3):
    db = [4] * N
    db[-1] = 3
    ks = [kernel_size] * N
    RF = [0] * N

    for b in range(N):
        if b == 0:
            RF[b] = 1 + db[b] * (ks[b] - 1) * (2**b)

        else:
            RF[b] = db[b] * (ks[b] - 1) * 2**b

    print(f"RF_i={RF}, RF_sum:{sum(RF)}")


def relative_error(x, y):
    return float((norm(x - y) / norm(y)).numpy())


def get_relative_error_stats(results, save_preds=False):
    """
    Returns list of tuples (pred, ture, relative_error)
    """
    flattened_pred = []
    flattened_y = []
    errors = []

    for batch in results:
        pred = batch[0]
        y = batch[1]
        x = batch[2]
        bound = batch[3]

        folder = Path(f"report/{pred.shape[2]}_preds")
        folder.mkdir(parents=True, exist_ok=True)

        for i, (sample_pred, sample_y, sample_x, sample_bound) in enumerate(
            zip(pred, y, x, bound)
        ):
            if save_preds and i < 10:
                np.savez(f"{folder}/pred_{i}_.npz", sample_pred.cpu().detach().numpy())
                np.savez(f"{folder}/y_{i}_.npz", sample_y.cpu().detach().numpy())
                np.savez(f"{folder}/x_{i}_.npz", sample_x.cpu().detach().numpy())
                np.savez(
                    f"{folder}/bound_{i}_.npz", sample_bound.cpu().detach().numpy()
                )
            errors.append(relative_error(sample_pred, sample_y))
            flattened_pred.append(sample_pred[0])
            flattened_y.append(sample_y[0])

    return flattened_pred, flattened_y, errors


def get_min_max_median_errors_idx(errors):
    argsor = np.argsort(errors)
    max_error_idx = argsor[0]
    min_error_idx = argsor[-1]
    median_error_idx = argsor[len(errors) // 2]

    return (max_error_idx, min_error_idx, median_error_idx)


def get_one_sample(results):
    idx = 4787
    pred = results[idx][0][0][0]
    y = results[idx][1][0][0]
    return pred, y
