# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union

from pcld.modules.distributions import DiagonalGaussianDistribution
from pcld.modules.losses.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class KLChamferDistance(nn.Module):
    def __init__(self,
                 kl_weight: float = 1.0,
                 chamfer_func: str = "ChamferDistanceL2"):

        super().__init__()

        self.kl_weight = kl_weight

        if chamfer_func == "ChamferDistanceL2":
            self.chamfer_criterion = ChamferDistanceL2()
        else:
            self.chamfer_criterion = ChamferDistanceL1()

    def forward(self,
                posteriors: Union[DiagonalGaussianDistribution, torch.distributions.Normal],
                pred_centers: torch.FloatTensor,
                centers: torch.FloatTensor,
                optimizer_idx: int,
                global_step: int,
                batch_idx: int,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            pred_centers (torch.FloatTensor): [B, M, 3]
            centers (torch.FloatTensor): [B, M, 3]
            optimizer_idx (int):
            global_step (int):
            batch_idx (int):
            split (str):
            **kwargs:

        Returns:

        """

        if isinstance(posteriors, DiagonalGaussianDistribution):
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            mu_ref = torch.zeros_like(posteriors.loc)
            scale_ref = torch.ones_like(posteriors.scale)
            standard_normal = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = torch.distributions.kl_divergence(posteriors, standard_normal).mean()

        chamfer_loss = self.chamfer_criterion(pred_centers.float(), centers.float())

        loss = kl_loss * self.kl_weight + chamfer_loss

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/chamfer".format(split): chamfer_loss.detach(),
            "{}/kl".format(split): kl_loss.detach(),
        }

        return loss, log


class ChamferDistanceOnly(nn.Module):
    def __init__(self,
                 chamfer_func: str = "ChamferDistanceL2"):

        super().__init__()

        # self.kl_weight = kl_weight
        self.chamfer_criterion = chamfer_func

        if chamfer_func == "ChamferDistanceL2":
            self.chamfer_criterion = ChamferDistanceL2()
        else:
            self.chamfer_criterion = ChamferDistanceL1()

    def forward(self,
                # posteriors: Union[DiagonalGaussianDistribution, torch.distributions.Normal],
                pred_centers: torch.FloatTensor,
                centers: torch.FloatTensor,
                optimizer_idx: int,
                global_step: int,
                batch_idx: int,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            pred_centers (torch.FloatTensor): [B, M, 3]
            centers (torch.FloatTensor): [B, M, 3]
            optimizer_idx (int):
            global_step (int):
            batch_idx (int):
            split (str):
            **kwargs:

        Returns:

        """

        # if isinstance(posteriors, DiagonalGaussianDistribution):
        #     kl_loss = posteriors.kl(dims=(1, 2))
        #     kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # else:
        #     mu_ref = torch.zeros_like(posteriors.loc)
        #     scale_ref = torch.ones_like(posteriors.scale)
        #     standard_normal = torch.distributions.Normal(mu_ref, scale_ref)
        #     kl_loss = torch.distributions.kl_divergence(posteriors, standard_normal).mean()

        chamfer_loss = self.chamfer_criterion(pred_centers.float(), centers.float())

        # loss = kl_loss * self.kl_weight + chamfer_loss
        loss = chamfer_loss

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/chamfer".format(split): chamfer_loss.detach(),
            # "{}/".format(split)+self.chamfer_criterion: chamfer_loss.detach(),
            # "{}/kl".format(split): kl_loss.detach(),
        }

        return loss, log