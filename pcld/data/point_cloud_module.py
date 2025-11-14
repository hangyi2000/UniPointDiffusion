# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl

from typing import Optional
from pcld.utils import instantiate_from_config
from .datasets.point_cloud import PointCloudDataset_multi_v405
from .utils import worker_init_fn

class PointCloudDataModule_multi_v405(pl.LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            dataset_folder: str,
            split_folder: str,
            pc_size: int = 2048,
            surface_sampling: bool = True,
            transform: Optional[dict] = None,

            batch_size: int = 1,
            num_workers: int = 4,
            only_get_16_category = False
    ):

        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder
        self.split_folder = split_folder
        self.pc_size = pc_size
        self.surface_sampling = surface_sampling
        self.transform = transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.only_get_16_category = only_get_16_category

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            self.train_dataset = PointCloudDataset_multi_v405(
                self.dataset_name, self.dataset_folder, self.split_folder, "train",
                transform=None, surface_sampling=True,
                pc_size=2048, surfaces_folder="surfaces"
            )

            self.val_dataset = PointCloudDataset_multi_v405(
                self.dataset_name, self.dataset_folder, self.split_folder, "val",
                transform=None, surface_sampling=True,
                pc_size=2048, surfaces_folder="surfaces"
            )

            self.test_dataset = PointCloudDataset_multi_v405(
                self.dataset_name, self.dataset_folder, self.split_folder, "test",
                transform=None, surface_sampling=True,
                pc_size=2048, surfaces_folder="surfaces"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )
