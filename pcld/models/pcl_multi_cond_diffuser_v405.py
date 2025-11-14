# -*- coding: utf-8 -*-

import inspect
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import repeat
import numpy as np

from pcld.modules.denoisers.scheduling_ddim import DDIMScheduler
from pcld.modules.denoisers.scheduling_ddpm import DDPMScheduler
from pcld.modules.conditional_encoders.class_encoder import ClassEmbedder
from pcld.modules.conditional_encoders.context_encoder import MultiEncoder
from pcld.lr_scheduler import BaseScheduler
from pcld.utils import instantiate_from_config

from PIL import Image
import io
import base64

from pcld.modules.losses.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from ext.emd.emd_module import emdModule

category2text_img = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "video_display",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "watercraft",
}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class PointCloudLatentDiffuser(pl.LightningModule):

    def __init__(self, *,
                 vae_cfg,
                 context_cfg,
                 class_encoder_cfg,
                 denoiser_cfg,
                 scheduler_cfg,
                 optim_cfg,
                 loss_cfg,
                 scale_by_std: bool = False,
                 z_scale_factor: float = 1.0,
                 context_trainable: bool = False,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        # 1. vae stage
        vae = instantiate_from_config(vae_cfg)
        self.vae = vae.eval()
        self.vae.train = disabled_train
        for param in self.vae.parameters():
            param.requires_grad = False

        self.latent_dim = self.vae.latent_dim

        # 2. conditional encoder stage
        self.context_cfg = context_cfg
        self.context_trainable = context_trainable
        context_encoder: MultiEncoder = instantiate_from_config(context_cfg)
        if not context_trainable:
            context_encoder = context_encoder.eval()
            for param in context_encoder.parameters():
                param.requires_grad = False

        self.context_encoder = context_encoder

        self.class_encoder_cfg = class_encoder_cfg
        class_encoder: ClassEmbedder = instantiate_from_config(class_encoder_cfg)
        self.class_encoder = class_encoder

        # 2. denoisers
        self.denoiser: nn.Module = instantiate_from_config(denoiser_cfg)
        self.optim_cfg = optim_cfg

        # 3. scheduling strategy
        self.scheduler_cfg = scheduler_cfg

        self.guidance_scale = scheduler_cfg.GUIDANCE_SCALE
        self.guidance_uncodp = scheduler_cfg.GUIDANCE_UNCONDP

        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        self.predict_epsilon = scheduler_cfg.get("PREDICT_EPSILON", True)

        self.noise_scheduler: DDPMScheduler = self.get_scheduler(scheduler_cfg.noise)
        self.denoise_scheduler: DDIMScheduler = self.get_scheduler(scheduler_cfg.denoise)

        # 4. loss configures
        self.loss_cfg = loss_cfg

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        from pcld.utils.visualizers.pythreejs_viewer import PyThreeJSViewer
        self.viewer = PyThreeJSViewer(settings={}, render_mode="WEBSITE")

        self.ground_truth_point_cloud = []
        self.generated_point_cloud = []

        # chamfer distance
        self.loss_cd = ChamferDistanceL2()

    def loss_emd(self, x1, x2):
        emd = emdModule()
        dis, assigment = emd(x1, x2, 0.002, 10000) # 0.005, 50 for training 
        assigment = assigment.cpu().numpy()
        assigment = np.expand_dims(assigment, -1)
        x2 = np.take_along_axis(x2, assigment, axis = 1)
        d = (x1 - x2) * (x1 - x2)
        d = np.sqrt(d.cpu().sum(-1)).mean()
        return d

    def get_scheduler(self, cfg):
        scheduler_type = cfg.TYPE.lower()
        if scheduler_type == "ddpm":
            return DDPMScheduler(
                num_train_timesteps=cfg.NUM_TRIAN_STEPS,
                beta_start=cfg.BETA_START,
                beta_end=cfg.BETA_END,
                beta_schedule=cfg.BETA_SCHEDULE,
                variance_type=cfg.VARIANCE_TYPE,
                clip_sample=cfg.CLIP_SAMPLE,
            )
        elif scheduler_type == "ddim":
            return DDIMScheduler(
                num_train_timesteps=cfg.NUM_TRIAN_STEPS,
                beta_start=cfg.BETA_START,
                beta_end=cfg.BETA_END,
                beta_schedule=cfg.BETA_SCHEDULE,
                clip_sample=cfg.CLIP_SAMPLE,
                set_alpha_to_one=cfg.SET_ALPHA_TO_ONE,
                steps_offset=cfg.STEPS_OFFSET,
            )
        else:
            raise NotImplementedError("Only support ddpm, ddim schdulers now")

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

    @property
    def disable_prog(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return not zero_rank

    def configure_optimizers(self) -> Tuple[List, List]:

        trainable_parameters = list(self.denoiser.parameters())

        # if the conditional encoder is trainable
        if self.context_trainable:
            trainable_parameters += list(self.context_encoder.parameters())
        trainable_parameters += list(self.class_encoder.parameters())

        lr = self.learning_rate
        betas = self.optim_cfg.get("betas", (0.9, 0.999))
        weight_decay = self.optim_cfg.get("weight_decay", 1e-2)
        optimizers = [torch.optim.AdamW(trainable_parameters, lr=lr, betas=betas, weight_decay=weight_decay)]
        schedulers = []

        if "scheduler_cfg" in self.optim_cfg:
            scheduler: BaseScheduler = instantiate_from_config(self.optim_cfg.scheduler_cfg)

            schedulers = [
                {
                    "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                } for optimizer in optimizers
            ]

        return optimizers, schedulers

    def latent_diffusion_forward(self, denoiser, latents, encoder_hidden_states, **kwargs):
        """
            heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

        Args:
            denoiser (nn.Module):
            latents (torch.FloatTensor): [bs, n_token, latent_dim]
            encoder_hidden_states (torch.FloatTensor): [bs, n_context_token, context_dim]

        Returns:
            outputs (dict): the forward diffusing outputs, and contain:
                - noise (torch.FloatTensor):
                - noise_pred (torch.FloatTensor):

        """

        with torch.no_grad():
            # Sample noise that we"ll add to the latents
            # [batch_size, n_token, latent_dim]
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            # Sample a random timestep for each motion
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # # Predict the noise residual
        noise_pred = denoiser(
            x=noisy_latents,
            timesteps=timesteps,
            y=encoder_hidden_states
        )
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.loss_cfg.elbo_weight != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        outputs = {
            "x_0": latents,
            "noise": noise,
            "noise_pred": noise_pred,
            "noise_prior": noise_prior,
            "noise_pred_prior": noise_pred_prior,
        }

        return outputs

    def latent_diffusion_reverse(self, denoiser, encoder_hidden_states, noise_dim,
                                 desc: str = "Diffusion Reverse:", **kwargs):
        bsz = encoder_hidden_states[1].shape[0]
        latents = torch.randn(
            (bsz, *noise_dim),
            device=encoder_hidden_states[1].device,
            dtype=encoder_hidden_states[1].dtype,
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.denoise_scheduler.init_noise_sigma
        # set timesteps
        self.denoise_scheduler.set_timesteps(self.scheduler_cfg.denoise.NUM_INFERENCE_STEPS)
        timesteps = self.denoise_scheduler.timesteps.to(encoder_hidden_states[1].device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.denoise_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.scheduler_cfg.denoise.ETA

        # reverse
        for i, t in enumerate(tqdm(timesteps, disable=self.disable_prog, desc=desc, leave=False)):
            latent_model_input = latents
            t = torch.tensor([t]).to(t.device)
            noise_pred = denoiser(
                x=latent_model_input, 
                timesteps=t, 
                y=encoder_hidden_states
                )
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_text = noise_pred
                noise_pred_uncond = denoiser(
                    x=latent_model_input,
                    timesteps=t,
                    y=encoder_hidden_states,
                    force_mask=True
                )
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.denoise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        return latents
    
    def diffusion_forward(self,
                          class_num,
                          img,
                          text,
                          z_q: torch.FloatTensor):
        """ Forward Diffusion in Latents.

                Args:
                    - text (list of str):
                    - z_q (torch.FloatTensor): [n_pts, c]

                Returns:

                """

        with torch.no_grad():
            img_emb, text_emb = self.context_encoder(img, text)
        class_num_emb = self.class_encoder(class_num)    # [B, latent_dim]
        class_num_emb = class_num_emb.unsqueeze(1)  # [B, 1, latent_dim]
        outputs = self.latent_diffusion_forward(self.denoiser, z_q, (class_num_emb, img_emb, text_emb))

        return outputs

    def diffusion_reverse(self, cond):
        class_num = cond[0]
        img = cond[1]
        text = cond[2]

        with torch.no_grad():
            img_emb, text_emb = self.context_encoder(img, text)
            class_num_emb = self.class_encoder(class_num)
            class_num_emb = class_num_emb.unsqueeze(1)  # [B, 1, latent_dim]

            z_q = self.latent_diffusion_reverse(
                self.denoiser, (class_num_emb, img_emb, text_emb), self.latent_dim,
                desc="Latent Diffusion Reverse:"
            )

        return z_q
    
    def diffusion_reverse_test(self, cond):
        class_num = cond[0]
        img = cond[1]
        text = cond[2]

        with torch.no_grad():
            img_emb, text_emb = self.context_encoder(img, text)
            img_emb = 0.5 * img_emb

            class_num_emb = self.class_encoder(class_num)
            class_num_emb = class_num_emb.unsqueeze(1)  # [B, 1, latent_dim]

            z_q = self.latent_diffusion_reverse(
                self.denoiser, (class_num_emb, img_emb, text_emb), self.latent_dim,
                desc="Latent Diffusion Reverse:"
            )
            z_q_class_num = self.latent_diffusion_reverse(
                self.denoiser, (class_num_emb, torch.zeros_like(img_emb), torch.zeros_like(text_emb)), self.latent_dim,
                desc="Latent Diffusion Reverse:"
            )
            z_q_img = self.latent_diffusion_reverse(
                self.denoiser, (torch.zeros_like(class_num_emb), img_emb, torch.zeros_like(text_emb)), self.latent_dim,
                desc="Latent Diffusion Reverse:"
            )
            z_q_text = self.latent_diffusion_reverse(
                self.denoiser, (torch.zeros_like(class_num_emb), torch.zeros_like(img_emb), text_emb), self.latent_dim,
                desc="Latent Diffusion Reverse:"
            )

        return z_q, z_q_class_num, z_q_img, z_q_text

    def forward(self, batch):

        surface_pc = batch["surface_pc"]
        surface_feats = batch["surface_feats"] if "surface_feats" in batch else None
        text = batch['text']
        img = batch['image']
        class_num = batch['class_num']
        img = img.to(surface_pc.device)
        class_num = class_num.to(surface_pc.device)

        z_q, centers = self.encode_first_stage(surface_pc, surface_feats)

        return self.diffusion_forward(class_num, img, text, z_q)

    @torch.no_grad()
    def encode_first_stage(self,
                           surface_pc: torch.FloatTensor,
                           surface_feats: Optional[torch.FloatTensor] = None):
        z_q, fps_center_pos, posterior = self.vae.encode(surface_pc)
        z_q = self.z_scale_factor * z_q

        return z_q, fps_center_pos

    @torch.no_grad()
    def decode_first_stage(self, z_q: torch.FloatTensor):

        z_q = 1. / self.z_scale_factor * z_q
        surface_latents, center_pos = self.vae.decode(z_q)

        return surface_latents, center_pos

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
                and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            surface_pc = batch["surface_pc"]
            surface_feats = batch["surface_feats"] if "surface_feats" in batch else None
            z_q, centers = self.encode_first_stage(surface_pc=surface_pc, surface_feats=surface_feats)
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def compute_loss(self, outputs, split):
        """

        Args:
            outputs (dict):
                - x_0:
                - noise:
                - noise_prior:
                - noise_pred:
                - noise_pred_prior:

            split (str):

        Returns:

        """

        if self.predict_epsilon:
            simple_loss = F.mse_loss(outputs["noise_pred"], outputs["noise"])
        else:
            simple_loss = F.mse_loss(outputs["noise_pred"], outputs["x_0"])

        loss = simple_loss

        if self.loss_cfg.elbo_weight > 0:
            elbo_loss = F.mse_loss(outputs["noise_pred_prior"], outputs["noise_prior"])
            loss += self.loss_cfg.elbo_weight * elbo_loss
        else:
            elbo_loss = 0.0

        loss_dict = {
            f"{split}/total_loss": loss.clone().detach(),
            f"{split}/simple": simple_loss.detach(),
        }

        if self.loss_cfg.elbo_weight > 0:
            loss_dict[f"f{split}/elbo"] = elbo_loss.detach()

        return loss, loss_dict

    def training_step(self, batch: Dict[str, Union[torch.FloatTensor, List[str]]],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface_pc (torch.FloatTensor): [n_pts, 4]
                - surface_feats (torch.FloatTensor): [n_pts, c]
                - text (list of str):

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        batch_size = len(batch['category'])

        # diffusion process return with noise and noise_pred
        forward_outputs = self(batch)

        loss, loss_dict = self.compute_loss(forward_outputs, "val")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                      batch_size=batch_size)

        self.log("global_step", float(self.global_step), prog_bar=False, logger=True, on_step=True, on_epoch=False,
                 batch_size=batch_size)

        if "scheduler_cfg" in self.optim_cfg:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Dict[str, torch.FloatTensor],
                        batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface_pc (torch.FloatTensor): [n_pts, 4]
                - surface_feats (torch.FloatTensor): [n_pts, c]
                - text (list of str):

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        # diffusion process return with noise and noise_pred
        forward_outputs = self(batch)

        loss, loss_dict = self.compute_loss(forward_outputs, "val")
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                batch_size=len(batch["category"]))

        # 改不可以
        return loss

    @torch.no_grad()
    def cond2points(self,
                    cond,
                    normalize: bool = False,
                    clamp_center: bool = False):
        z_q = self.diffusion_reverse(cond)

        # vae decoder
        latents, centers = self.decode_first_stage(z_q)

        if clamp_center:
            centers = centers.clamp_(-1., 1.)

        if normalize:
            min_v = centers.min()
            max_v = centers.max()
            centers = (centers - min_v) / (max_v - min_v) * 2 - 1

        outputs = {
            "centers": centers.cpu().numpy(),
            "latents": latents.cpu().numpy()
        }

        return outputs
    

    def cond2points_test(self, 
                         cond, 
                         normalize: bool = False, 
                         clamp_center: bool = False):
        z_q, z_q_class_num, z_q_img, z_q_text = self.diffusion_reverse_test(cond)

        # vae decoder
        latents, centers = self.decode_first_stage(z_q)

        if clamp_center:
            centers = centers.clamp_(-1., 1.)

        if normalize:
            min_v = centers.min()
            max_v = centers.max()
            centers = (centers - min_v) / (max_v - min_v) * 2 - 1

        outputs = {
            "centers": centers.cpu().numpy(),
            "latents": latents.cpu().numpy(),
        }

        return outputs
