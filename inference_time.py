import numpy as np
import torch

from UNet.lightning_modules import *
from UNet.models import *

model = UNetAvg2048Module300k(
    model_hparams={
        "epochs": 50,
        "batch_size": 16,
        "SHAPE": 1024,
        "N_SAMPLES": 5000,
        "filters": [4, 8, 16, 32, 64, 128, 256],
        "bilinear": True,
        "in_channels": 1,
        "out_channels": 1,
    },
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)

model.load_from_checkpoint("artifacts/2048300k/epoch=29-step=12150.ckpt")
device = torch.device("cuda")
model = model.model
model.to(device)
dummy_input = torch.randn(1, 1, 2048, 2048, dtype=torch.float).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)
repetitions = 300
timings = np.zeros((repetitions, 1))
# GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(curr_time)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)
print(std_syn)
