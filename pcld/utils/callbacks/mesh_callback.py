# -*- coding: utf-8 -*-

import os
import wandb
import numpy as np
from typing import Tuple, Generic, Dict, List, Union
from tqdm import tqdm

import torch
import torchvision
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_info
from pytorch_lightning.callbacks import Callback

from pcld.utils.visualizers.pythreejs_viewer import PyThreeJSViewer
from pcld.utils.visualizers import html_util

from PIL import Image
import io
import base64


def convert_visual_info_to_html(viewer, visuals_info_list, bbox_size):
    html_frame_list = []
    if len(visuals_info_list) < 1:
        return html_frame_list

    for visuals_info in visuals_info_list:
        centers = visuals_info["centers"]
        text = visuals_info["text"]

        for i, center in enumerate(centers):
            center[:, 0] += i * np.max(bbox_size)
            # viewer.add_points(center, c=(center + 1) / 2,
            #                   shading={"point_size": 0.2, "point_shape": "square"})
            viewer.add_points(center, c=(center + 1) / 2,
                              shading={"point_size": 0.01, "point_shape": "square"})

            if "vertices" in visuals_info:
                vertices = visuals_info["vertices"][i]
                faces = visuals_info["faces"][i]

                if vertices is not None:
                    vertices[:, 1] -= np.max(bbox_size)
                    vertices[:, 0] += i * np.max(bbox_size)
                    viewer.add_mesh(vertices, faces)

        content_html = viewer.to_html(html_frame=False)
        table_html = html_util.to_single_row_table(caption=text, content=content_html)
        html_frame = html_util.to_html_frame(table_html)
        html_frame_list.append(html_frame)

        viewer.reset()

    return html_frame_list

class MultiCondPointDiffuserLogger(Callback):
    def __init__(self,
                 step_frequency: int,
                 num_samples: int = 4,
                 sample_times: int = 4,
                 bounding_box: Union[List[float], Tuple[float]] = (-1, -1, -1, 1, 1, 1),
                 normalize: bool = False) -> None:

        super().__init__()

        self.bbox_min = np.array(bounding_box[0:3])
        self.bbox_max = np.array(bounding_box[3:6])
        self.bbox_size = self.bbox_max - self.bbox_min
        self.normalize = normalize

        self.step_freq = step_frequency
        self.num_samples = num_samples
        self.sample_times = sample_times
        self.has_train_logged = False
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }

        self.viewer = PyThreeJSViewer(settings={}, render_mode="WEBSITE")

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        # raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    def to_image_embed_tag(self, image: np.ndarray):

        # Convert np.ndarray to bytes
        img = Image.fromarray(image)
        raw_bytes = io.BytesIO()
        img.save(raw_bytes, "PNG")

        # Encode bytes to base64
        image_base64 = base64.b64encode(raw_bytes.getvalue()).decode("utf-8")

        image_tag = f"""
        <img src="data:image/png;base64,{image_base64}" alt="Embedded Image">
        """

        return image_tag

    @rank_zero_only
    def log_local(self, save_dir: str, split: str, visuals_info_list: List[Dict],
                  global_step: int, current_epoch: int, batch_idx: int,
                  prog_bar: bool = False) -> None:

        root = os.path.join(save_dir, "visuals", split)
        os.makedirs(root, exist_ok=True)

        for n_sample, visuals_info in enumerate(tqdm(visuals_info_list, desc="Saving Point Cloud:",
                                                     disable=not prog_bar, leave=False)):
            text = visuals_info["text"]
            img = visuals_info["image"]
            centers = visuals_info["centers"]

            # transform img from torch tensor to np.ndarray
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = np.uint8((img + 1) / 2 * 255)
            image_tag = self.to_image_embed_tag(img)

            # horizon: left is center points, and right is mesh
            for i, center in enumerate(centers):
                center[:, 0] += i * np.max(self.bbox_size)
                self.viewer.add_points(center, c=(center + 1) / 2,
                                       shading={"point_size": 0.2, "point_shape": "square"})
            content_html = self.viewer.to_html(html_frame=False)
            table_html = html_util.to_double_row_table(caption=text, content1=content_html, content2=image_tag)
            html_frame = html_util.to_html_frame(table_html)

            html_name = "gs-{:010}_e-{:06}_b-{:06}_s-{:03}.html".format(
                global_step, current_epoch,
                batch_idx, n_sample
            )
            with open(os.path.join(root, html_name), "w") as f:
                f.write(html_frame)

            self.viewer.reset()

    @rank_zero_only
    def log_local_test(self, save_dir: str, split: str, visuals_info_list: List[Dict],
                  global_step: int, current_epoch: int, batch_idx: int,
                  prog_bar: bool = False) -> None:

        root = os.path.join(save_dir, "visuals", split)
        os.makedirs(root, exist_ok=True)

        for n_sample, visuals_info in enumerate(tqdm(visuals_info_list, desc="Saving Point Cloud:",
                                                     disable=not prog_bar, leave=False)):
            text = visuals_info["text"]
            img = visuals_info["image"]
            centers = visuals_info["centers"]

            # transform img from torch tensor to np.ndarray
            img = img.cpu().numpy().transpose(1, 2, 0)
            # img = np.uint8((img + 1) / 2 * 255)
            img = np.uint8(img * 255)
            image_tag = self.to_image_embed_tag(img)

            for i, center in enumerate(centers):
                center[:, 1] -= 0 * np.max(self.bbox_size)
                center[:, 0] += i * np.max(self.bbox_size)
                self.viewer.add_points(center, c=(center + 1) / 2,
                                       shading={"point_size": 0.2, "point_shape": "square"})

            content_html = self.viewer.to_html(html_frame=False)
            table_html = html_util.to_double_row_table(caption=text, content1=content_html, content2=image_tag)
            html_frame = html_util.to_html_frame(table_html)

            html_name = "gs-{:010}_e-{:06}_b-{:06}_s-{:03}.html".format(
                global_step, current_epoch,
                batch_idx, n_sample
            )
            with open(os.path.join(root, html_name), "w") as f:
                f.write(html_frame)

            self.viewer.reset()

    def log_cond2points(self,
                        pl_module: pl.LightningModule,
                        batch: Dict[str, torch.FloatTensor],
                        batch_idx: int,
                        split: str = "train") -> None:
        """

        Args:
            pl_module:
            batch (dict): the batch sample information, and it contains:
                 - text (List[str]):
            batch_idx (int):
            split (str):

        Returns:

        """

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            # batch_size = len(batch["text"])
            batch_size = len(batch["category"])
            replace = batch_size < self.num_samples
            ids = np.random.choice(batch_size, self.num_samples, replace=replace)
            text = [batch["text"][i] for i in ids]
            img = [batch["image"][i] for i in ids]
            img = torch.stack(img, dim=0)
            class_num = [batch["class_num"][i] for i in ids]
            class_num = torch.stack(class_num, dim=0)

            sample_centers = []
            for i in range(self.sample_times):
                # outputs = pl_module.img2points(
                #     img,
                #     normalize=self.normalize,
                # )
                outputs = pl_module.cond2points(
                    (class_num, img, text),
                    normalize=self.normalize,
                )
                centers = outputs["centers"]
                sample_centers.append(centers)

            sample_centers = np.stack(sample_centers, axis=0)

        visuals_info_list = []
        for i in range(self.num_samples):
            visual_info = {
                "text": text[i],
                "image": img[i],
                "centers": sample_centers[:, i, :]
            }

            visuals_info_list.append(visual_info)

        self.log_local(pl_module.logger.save_dir, split, visuals_info_list,
                       pl_module.global_step, pl_module.current_epoch, batch_idx, prog_bar=True)

        if is_train:
            pl_module.train()

    def log_cond2points_test(self,
                        pl_module: pl.LightningModule,
                        batch: Dict[str, torch.FloatTensor],
                        batch_idx: int,
                        split: str = "train") -> None:
        """

        Args:
            pl_module:
            batch (dict): the batch sample information, and it contains:
                 - text (List[str]):
            batch_idx (int):
            split (str):

        Returns:

        """

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            text = batch["text"]
            img = batch["image"]
            class_num = batch["class_num"]
            model_name = batch["model_name"]
            img_path = batch["img_path"]

            batch_size = img.shape[0]
            # print("img_shape = ", img.shape)

            sample_centers = []

            for i in range(self.sample_times):
                outputs = pl_module(
                    (class_num, img, text),
                    normalize=self.normalize,
                )

                centers = outputs["centers"]
                sample_centers.append(centers)

            sample_centers = np.stack(sample_centers, axis=0)
            self.centers_list.append(sample_centers)

        visuals_info_list = []
        # for i in range(self.num_samples):
        for i in range(batch_size):
            visual_info = {
                "text": text[i]+model_name[i],
                "image": img[i],
                "centers": sample_centers[:, i, :],
            }

            visuals_info_list.append(visual_info)

        self.log_local_test(pl_module.logger.save_dir, split, visuals_info_list,
                       pl_module.global_step, pl_module.current_epoch, batch_idx, prog_bar=True)

        if is_train:
            pl_module.train()

    def check_frequency(self, step: int) -> bool:
        if step % self.step_freq == 0:
            return True
        return False

    def on_train_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                           outputs: Generic, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> None:

        if (self.check_frequency(pl_module.global_step) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "cond2points") and
                callable(pl_module.cond2points) and
                self.num_samples > 0):
            self.log_cond2points(pl_module, batch, batch_idx, split="train")
            self.has_train_logged = True

    def on_validation_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                                outputs: Generic, batch: Dict[str, torch.FloatTensor],
                                dataloader_idx: int, batch_idx: int) -> None:

        if self.has_train_logged:
            self.log_cond2points(pl_module, batch, batch_idx, split="val")
            self.has_train_logged = False

    def on_test_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                                outputs: Generic, batch: Dict[str, torch.FloatTensor],
                                dataloader_idx: int, batch_idx: int) -> None:
        self.log_cond2points_test(pl_module, batch, self.tmp_batch_idx, split="test")
        self.tmp_batch_idx = self.tmp_batch_idx + 1
        pass
