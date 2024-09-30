# U-Net Poisson Solver

This is the GitHub repository for the paper `A Hybrid Method for Solving the Two-Dimensional Poisson Equation: Combining U-Net and Conjugate Gradient Method`. The project implements a U-Net and hybrid approach for solving Poisson equation using deep learning.


## Table of Contents
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Dataset Generation](#dataset-generation)
- [Training U-Net](#train-u-net)
- [Results and Examples](#results-and-examples)

## Installation

After activating the virtual environment, you can install specific package requirements as follows:

```bash
pip install -r requirements.txt
```

## Prerequisites

This code uses:
- Python 3.10+
- PyTorch, PyTorch Lightning as core framework.
- WanDB for experiments tracking.
- Multigrid method from PyAMG for dataset generation.
- Conjugate gradient method from scipy
- Plotly for visualization.


## Dataset generation

For dataset generation, you can use `generate_data.py` script and specify:

- `-s`: shape of samples (one number) -> (n x n)
- `-n`: number of samples
- `-j`: number of jobs for parallel generation. 

To run generation use:
```bash
python3 generate_data.py -s {shape} -n {num_of_samples} -j {num_of_jobs}
```
Example:

```bash
python3 generate_data.py -s 64 -n 1000 -j 4
```

## Train U-Net

To train neural network, you should specify dataset folder in `run_train.py` script, then call 

```bash
python3 run_train.py
```

In the script, you can configure different U-Net models and adjust hyperparameters.

Prebuilt torch models for different dataset shapes are in UNet/models.py, and lightning modules are in UNet/lightning_modules.py.

To test trained model you can use `run_test.py ` or simply `test.py` with specifying model checkpoint. 

## Results and examples