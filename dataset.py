import glob
import re
from typing import Callable

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class PoissonDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_array = np.load(self.data_paths[idx])
        x = data_array["b"]
        y = data_array["x"]
        bound = data_array["bound"]
        if self.transform:
            y = self.transform(y)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        return torch.tensor(x), torch.tensor(y), bound


class PoissonDatasetResize(Dataset):
    def __init__(self, data_paths, transform=None, shape=1024):
        self.data_paths = data_paths
        self.transform = transform
        self.shape = shape

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_array = np.load(self.data_paths[idx])
        x = torch.tensor(data_array["b"])
        y = torch.tensor(data_array["x"])

        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)

        if self.transform:
            y = self.transform(y)

        return x, y


class PoissonDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform: Callable = lambda x: x * 100,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.data_shape = re.findall("[0-9]+", str(data_dir))[
            0
        ]  # first number in folder name = data_shape

    def setup(self, stage=None):
        data_paths = glob.glob(f"{self.data_dir}/*")

        train_files, test_files = train_test_split(
            data_paths, test_size=0.1, random_state=42
        )
        train_files, val_files = train_test_split(
            train_files, test_size=0.1, random_state=42
        )
        test_files = data_paths

        self.train_dataset = PoissonDataset(train_files, self.transform)
        self.val_dataset = PoissonDataset(val_files, self.transform)

        self.test_dataset = PoissonDataset(test_files, self.transform)  # TODO

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ResizeDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform: Callable = lambda x: x * 100,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.data_shape = re.findall("[0-9]+", str(data_dir))[
            0
        ]  # first number in folder name = data_shape

    def setup(self, stage=None):
        data_paths = glob.glob(f"{self.data_dir}/*")

        test_files = data_paths

        self.test_dataset = PoissonDatasetResize(test_files, self.transform)

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
