# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from einops import repeat
from tqdm import tqdm
from typing import Union

from pcld.utils import instantiate_from_config

from pcld.modules.pcl_vae_module import SurfaceEncoder, SurfaceDecoder

from pcld.modules.losses.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from ext.emd.emd_module import emdModule

import numpy as np
category2text = {
    "02691156": "airplane",
    "02747177": "trash bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}


def save_to_txt(tosave, fname):
    if torch.is_tensor(tosave):
        tosave = tosave.squeeze().cpu().numpy()

    np.savetxt(fname, tosave, delimiter=" ")   #.xyz


class PointLatentKLModel(pl.LightningModule):

    def __init__(self, *,
                 dim: int = 768,  # transformer dimension
                 num_centers: int = 1024,
                 num_freqs: int = 8,
                 mask_ratio: float = 0.15,
                 encoder_cfg,
                 decoder_cfg,
                 loss_cfg,
                 scheduler_cfg: Optional[OmegaConf] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        self.encoder = SurfaceEncoder(
            dim=dim,
            num_centers=num_centers,
            num_freqs=num_freqs,
            **encoder_cfg
        )

        self.decoder = SurfaceDecoder(
            dim=dim,
            num_centers=num_centers,
            **decoder_cfg
        )

        self.mask_ratio = mask_ratio
        self.loss = instantiate_from_config(loss_cfg)

        self.scheduler_cfg = scheduler_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.latent_dim = self.encoder.latent_dim
        self.num_centers = num_centers

        self.save_hyperparameters()

        from pcld.utils.visualizers.pythreejs_viewer import PyThreeJSViewer
        self.viewer = PyThreeJSViewer(settings={}, render_mode="WEBSITE")

        self.loss_cd_l1 = ChamferDistanceL1()
        self.loss_cd_l2 = ChamferDistanceL2()

    def loss_emd(self, x1, x2):
        emd = emdModule()
        dis, assigment = emd(x1, x2, 0.002, 10000)
        assigment = assigment.cpu().numpy()
        assigment = np.expand_dims(assigment, -1)
        x2 = np.take_along_axis(x2, assigment, axis = 1)
        d = (x1 - x2) * (x1 - x2)
        d = np.sqrt(d.cpu().sum(-1)).mean()
        return d

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = list(self.encoder.parameters()) + list(self.decoder.parameters())

        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        schedulers = []

        if self.scheduler_cfg is not None:
            scheduler = instantiate_from_config(self.scheduler_cfg)

            schedulers = [
                {
                    "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                } for optimizer in optimizers
            ]

        return optimizers, schedulers

    def check_use_gt_center(self):
        return self.use_gt_center

    def encode(self, surface_pc: torch.FloatTensor):
        """

        Args:
            surface_pc (torch.FloatTensor): [B, N, 3];

        Returns:
            - latent (torch.FloatTensor):
            - center_pos (torch.FloatTensor):
            - posterior (torch.distribution.Normal):
        """

        # [B x C], [torch.distribution.Normal]
        latent, center_pos, posterior = self.encoder.forward(pc=surface_pc)

        return latent, center_pos, posterior

    def decode(self, z_q):
        """

        Args:
            z_q (torch.FloatTensor):

        Returns:
            center_latents (torch.FloatTensor): [B, T, C]
            center_pos (torch.FloatTensor): [B, T, 3]
        """

        center_latents, center_pos = self.decoder.decode_center_latents(z_q)

        return center_latents, center_pos

    def forward(self,
                surface_pc: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            surface_pc (torch.FloatTensor):
            sample_posterior (bool):

        Returns:
            - logits (torch.FloatTensor): [bs, n_samples, c_o]
            - qloss (torch.FloatTensor):
            - posterior (DiagonalGaussianDistribution):

        """

        # ipdb.set_trace()
        latent, center_pos, posterior = self.encode(surface_pc)
        center_latents, pred_centers = self.decoder.decode_center_latents(latent)

        outputs = {
            "latent": latent,
            "posterior": posterior,
            "center_pos": center_pos,
            "pred_centers": pred_centers
        }

        return outputs

    @property
    def disable_prog(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return not zero_rank

    @torch.no_grad()
    def reconstruct(self, surface_pc: torch.FloatTensor):

        latent, center_pos, posterior = self.encode(surface_pc)

        center_latents, center_pred = self.decode(latent)

        outputs = {
            "centers": center_pos.cpu().numpy(),
            "pred_centers": center_pred.cpu().numpy(),
            "center_latents": center_latents.cpu().numpy(),
        }

        return outputs

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - field_pts (torch.FloatTensor): [n_pts, 4]
                - field_feats (torch.FloatTensor): [n_pts, c]
                - points (torch.FloatTensor): [bs, n_samples, 3]
                - labels (torch.LongTensor): [bs, n_samples]

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        surface_pc = batch["surface_pc"]
        surface_feats = batch["surface_feats"] if "surface_feats" in batch else None
        batch_size = surface_pc.shape[0]

        outputs = self(surface_pc, surface_feats)

        # autoencoder
        loss, log_dict_ae = self.loss(outputs["posterior"],
                                      outputs["pred_centers"],
                                      outputs["center_pos"],
                                      optimizer_idx, self.global_step, batch_idx, split="train")

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_epoch=True,
                      batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface_pc = batch["surface_pc"]
        surface_feats = batch["surface_feats"] if "surface_feats" in batch else None
        batch_size = surface_pc.shape[0]

        outputs = self(surface_pc, surface_feats)
        loss, log_dict_ae = self.loss(outputs["posterior"],
                                      outputs["pred_centers"],
                                      outputs["center_pos"],
                                      0, self.global_step, batch_idx, split="val")

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=False,
                      on_epoch=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def on_test_start(self) -> None:

        self.ground_truth_point_cloud = []
        self.generated_point_cloud = []
        self.latent_codes = []
        self.class_num = []

    def test_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface_pc = batch["surface_pc"]
        surface_feats = batch["surface_feats"] if "surface_feats" in batch else None
        class_num = batch["class"]
        batch_size = surface_pc.shape[0]

        outputs = self(surface_pc, surface_feats)
        latent, center_pos, posterior = self.encoder.forward(pc=surface_pc)

        self.ground_truth_point_cloud.append(surface_pc.cpu().numpy())
        self.generated_point_cloud.append(outputs["pred_centers"].cpu().numpy())
        self.latent_codes.append(latent.cpu().numpy())
        self.class_num.append(class_num.cpu().numpy())

    def on_test_end(self) -> None:       
        generated_point_cloud = np.concatenate(self.generated_point_cloud, axis=0)
        ground_truth_point_cloud = np.concatenate(self.ground_truth_point_cloud, axis=0)
        latent_codes = np.concatenate(self.latent_codes, axis=0)
        class_num = np.concatenate(self.class_num, axis=0)
        np.savez("./experiments/point_vae/generated_point_cloud.npz", generated_point_cloud)
        np.savez("./experiments/point_vae/ground_truth_point_cloud.npz", ground_truth_point_cloud)
        np.savez("./experiments/point_vae/latent_codes.npz", latent_codes)
        np.savez("./experiments/point_vae/class_num.npz", class_num)
