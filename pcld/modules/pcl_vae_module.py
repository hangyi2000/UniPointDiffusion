# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps, knn
from torch_scatter import scatter_max
from torch.nn.utils import weight_norm
from typing import Optional, Union, List, Tuple
from einops import repeat

from pcld.modules.distributions import DiagonalGaussianDistribution
from pcld.modules.transformer import NormLayer, MultiHeadAttention, FeedForward, init_weights
from pcld.modules.embedder import FourierEmbedder


class PointConv(nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, feats=None, embedder=None):
        row, col = edge_index

        # pos: b x M x 3
        # pos_dst: b x M x k x 3 ->

        # out: b x M x k x 3
        out = pos[row] - pos_dst[col]

        if embedder is not None:
            out = embedder(out)

        if feats is not None:
            # out: b x M x k x c1
            feats = feats[row]
            out = torch.cat([out, feats], dim=-1)

        # local_out: b x M x k x c_local
        if self.local_nn is not None:
            out = self.local_nn(out)

        # b x M x c_local
        out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        # b x M x c_global
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


class PointNet(nn.Module):
    def __init__(self, *,
                 num_centers,
                 in_channels: int,
                 out_channels: int,
                 top_k: int = 32,
                 hidden_dim: int = 256,
                 feat_dim: int = 0):
        super().__init__()

        self.num_centers = num_centers
        self.top_k = top_k

        self.conv = PointConv(
            local_nn=nn.Sequential(
                weight_norm(nn.Linear(in_channels + feat_dim, hidden_dim)),
                nn.ReLU(True),
                weight_norm(nn.Linear(hidden_dim, hidden_dim))
            ),
            global_nn=nn.Sequential(
                weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(True),
                weight_norm(nn.Linear(hidden_dim, out_channels))
            )
        )

    def forward(self,
                pc: torch.FloatTensor,
                feats: Optional[torch.FloatTensor] = None,
                embedder: Optional[nn.Module] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            embedder (nn.Module or None)

        Returns:
            center_feats (torch.FloatTensor): [B, M, c]
            center_pos (torch.FloatTensor): [B, M, 3]

        """

        # pc: B x N x 3
        B, N, D = pc.shape
        ratio = self.num_centers / N

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        # [B * N, 3]
        flatten_pos = pc.view(B * N, D)
        idx = fps(flatten_pos, batch, ratio=ratio)  # 0.0625

        center_pos = flatten_pos[idx]
        center_batch = batch[idx]

        # Finds for each element in `center_pos` the `k` nearest points in `pos_flatten`
        row, col = knn(flatten_pos, center_pos, self.top_k, batch, center_batch)
        edge_index = torch.stack([col, row], dim=0)

        if feats is not None:
            feats = feats.view(B * N, -1)

        center_feats = self.conv(flatten_pos, center_pos, edge_index, feats=feats, embedder=embedder)
        center_feats = center_feats.view(B, -1, center_feats.shape[-1])
        center_pos = center_pos.view(B, -1, 3)

        return center_feats, center_pos


class TransformerEncoder(nn.Module):
    def __init__(self,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 attn_drop_prob: float = 0.0,
                 proj_drop_prob: float = 0.0,
                 drop_path_prob: float = 0.1,
                 pre_norm: bool = False, **ignore_kwargs) -> None:

        super().__init__()

        self.pos_drop = nn.Dropout(p=proj_drop_prob)

        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, depth)]  # stochastic depth decay rule
        for idx in range(depth):
            layer = nn.ModuleList([
                NormLayer(dim, MultiHeadAttention(dim, heads=heads, dim_head=dim_head,
                                                  attn_drop=attn_drop_prob, proj_drop=proj_drop_prob),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm),
                NormLayer(dim, FeedForward(dim, mlp_dim),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm)]
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

        self.apply(init_weights)

    def forward(self, x: torch.FloatTensor, pos_embed: torch.FloatTensor) -> torch.FloatTensor:

        x += pos_embed
        x = self.pos_drop(x)

        for attn, ff in self.layers:
            # residual skip connection has been absorbed into attn and ff
            x = attn(x)
            x = ff(x)

        # if pre_norm is True, otherwise it is Identity.
        x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 dim: int,
                 memory_dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 attn_drop_prob: float = 0.0,
                 proj_drop_prob: float = 0.0,
                 drop_path_prob: float = 0.1,
                 pre_norm: bool = False, **ignore_kwargs):

        super().__init__()

        self.pos_drop = nn.Dropout(p=proj_drop_prob)

        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, depth)]  # stochastic depth decay rule
        for idx in range(depth):
            self_attn = nn.ModuleList([
                NormLayer(dim, MultiHeadAttention(dim, heads=heads, dim_head=dim_head,
                                                  attn_drop=attn_drop_prob, proj_drop=proj_drop_prob),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm),
                NormLayer(dim, FeedForward(dim, mlp_dim),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm)]
            )
            cross_attn = nn.ModuleList([
                NormLayer(dim, MultiHeadAttention(dim, context_dim=memory_dim, heads=heads, dim_head=dim_head,
                                                  attn_drop=attn_drop_prob, proj_drop=proj_drop_prob),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm),
                NormLayer(dim, FeedForward(dim, mlp_dim),
                          drop_path_prob=dpr[idx], pre_norm=pre_norm)]
            )

            layer = nn.ModuleList([self_attn, cross_attn])
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

        self.apply(init_weights)

    def forward(self,
                queries: torch.FloatTensor,
                memory: torch.FloatTensor,
                pos_embed: Union[torch.FloatTensor] = None) -> torch.FloatTensor:

        if pos_embed is not None:
            queries += pos_embed

        x = self.pos_drop(queries)

        for (self_attn_attn, self_attn_ffn), (cross_attn_attn, cross_attn_ffn) in self.layers:
            # residual skip connection has been absorbed into attn and ff
            x = self_attn_attn(x)
            x = self_attn_ffn(x)

            x = cross_attn_attn(x, context=memory)
            x = cross_attn_ffn(x)

        # if pre_norm is True, otherwise it is Identity.
        x = self.norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self, *,
                 num_centers,
                 dim=256,
                 num_freqs: int = 8,
                 pointnet_cfg,
                 transformer_cfg):
        super().__init__()

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs)

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        hidden_dim = pointnet_cfg.get("hidden_dim", 256)
        feat_dim = pointnet_cfg.get("feat_dim", 0)

        self.num_centers = num_centers
        self.top_k = pointnet_cfg.get("top_k", 32)

        self.conv = PointConv(
            local_nn=nn.Sequential(
                weight_norm(nn.Linear(self.fourier_embedder.out_dim + feat_dim, hidden_dim)),
                nn.ReLU(True),
                weight_norm(nn.Linear(hidden_dim, hidden_dim))
            ),
            global_nn=nn.Sequential(
                weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(True),
                weight_norm(nn.Linear(hidden_dim, dim))
            )
        )

        self.linear_embedder = nn.Linear(self.fourier_embedder.out_dim, dim)
        self.transformer = TransformerEncoder(dim, **transformer_cfg)

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            center_latents (torch.FloatTensor):
            center_pos (torch.FloatTensor):

        """

        # pc: B x N x 3
        B, N, D = pc.shape
        ratio = self.num_centers / N

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        # [B * N, 3]
        flatten_pos = pc.view(B * N, D)
        idx = fps(flatten_pos, batch, ratio=ratio)  # 0.0625

        center_pos = flatten_pos[idx]
        center_batch = batch[idx]

        # Finds for each element in `center_pos` the `k` nearest points in `pos_flatten`
        row, col = knn(flatten_pos, center_pos, self.top_k, batch, center_batch)
        edge_index = torch.stack([col, row], dim=0)

        if feats is not None:
            feats = feats.view(B * N, -1)

        x = self.conv(flatten_pos, center_pos, edge_index, feats=feats, embedder=self.fourier_embedder)

        x = x.view(B, -1, x.shape[-1])
        center_pos = center_pos.view(B, -1, 3)

        embeddings = self.linear_embedder(self.fourier_embedder(center_pos))
        center_latents = self.transformer(x, embeddings)

        return center_latents, center_pos


class ImplicitNetwork(nn.Module):
    def __init__(self, *,
                 in_channels: int = 384,
                 out_channels: int = 1,
                 hidden_dim: int = 512,
                 depth: int = 8,
                 skips: Union[List[int], Tuple[int]] = (4,),
                 skips_legacy: bool = True,
                 activate: str = "relu",
                 use_weight_norm: bool = True, **ignore_kwargs):

        super().__init__()

        self.depth = depth
        self.skips = skips
        self.skips_legacy = skips_legacy and hidden_dim > in_channels

        block_in = in_channels
        block_out = hidden_dim
        self.layers = nn.ModuleList([])
        for i in range(depth):
            linear = nn.Linear(block_in, block_out)
            if use_weight_norm:
                linear = weight_norm(linear)

            act_fn = self.get_activate_function(activate)
            self.layers.append(nn.Sequential(linear, act_fn))

            if (i + 1) in self.skips:
                if self.skips_legacy:
                    # IDR: https://github.com/lioryariv/idr/blob/main/code/model/implicit_differentiable_renderer.py#L38
                    block_in = hidden_dim
                    block_out = hidden_dim - in_channels
                else:
                    # NeRF: https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L80
                    block_in = hidden_dim + in_channels
                    block_out = hidden_dim
            else:
                block_in = hidden_dim
                block_out = hidden_dim

        linear = nn.Linear(block_out, out_channels)
        if use_weight_norm:
            linear = weight_norm(linear)

        self.layers.append(linear)

    def forward(self, inputs):

        h = inputs
        if self.skips_legacy:
            for i in range(self.depth):
                # print(i, self.layers[i][0], h.shape)
                h = self.layers[i](h)
                if i in self.skips:
                    h = torch.cat([inputs, h], dim=-1)
        else:
            for i in range(self.depth):
                if i in self.skips:
                    h = torch.cat([inputs, h], dim=-1)
                h = self.layers[i](h)

        h = self.layers[-1](h)

        return h

    @staticmethod
    def get_activate_function(activate):
        if activate == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activate == "softplus":
            act_fn = nn.Softplus(beta=100)
        else:
            raise ValueError(f"{activate} does not support yet.")

        return act_fn


class SurfaceEncoder(nn.Module):

    def __init__(self, *,
                 num_centers: int,
                 num_latents: int = 1,
                 top_k: int = 32,
                 num_freqs: int = 8,
                 hidden_dim: int = 256,
                 feat_dim: int = 0,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 attn_drop_prob: float = 0.0,
                 proj_drop_prob: float = 0.0,
                 drop_path_prob: float = 0.1):

        super().__init__()

        self.num_centers = num_centers
        self.num_latents = num_latents
        self.fourier_embeddings = FourierEmbedder(num_freqs=num_freqs)

        self.pointnet = PointNet(
            num_centers=num_centers,
            in_channels=self.fourier_embeddings.out_dim,
            out_channels=dim,
            top_k=top_k,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim
        )

        # surface encoder
        if num_latents == 1:
            self.mu_token = nn.Parameter(torch.randn(dim, dtype=torch.float32))
            self.logvar_token = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        else:
            self.mu_token = nn.Parameter(torch.randn(num_latents, dim, dtype=torch.float32))
            self.logvar_token = nn.Parameter(torch.randn(num_latents, dim, dtype=torch.float32))
        self.center_proj = nn.Linear(self.fourier_embeddings.out_dim, dim)
        self.encoder_pos_embed = nn.Parameter(torch.randn((num_latents * 2 + num_centers, dim), dtype=torch.float32))

        self.transformer_encoder = TransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            drop_path_prob=drop_path_prob,
            pre_norm=True
        )

        self.latent_dim = (num_latents, dim)

    def forward(self,
                pc: torch.FloatTensor,
                feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            center_latents (torch.FloatTensor):
            center_pos (torch.FloatTensor):

        """

        bs = pc.shape[0]

        # [B, M, c], [B, M, 3]
        center_feats, center_pos = self.pointnet(pc, feats, embedder=self.fourier_embeddings)
        centers_pe = self.center_proj(self.fourier_embeddings(center_pos))

        if self.num_latents == 1:
            mu_token = repeat(self.mu_token, "c -> b m c", b=bs, m=self.num_latents)
            logvar_token = repeat(self.logvar_token, "c -> b m c", b=bs, m=self.num_latents)
        else:
            mu_token = repeat(self.mu_token, "m c -> b m c", b=bs)
            logvar_token = repeat(self.logvar_token, "m c -> b m c", b=bs)

        x = torch.cat([mu_token, logvar_token, center_feats + centers_pe], dim=1)
        x = self.transformer_encoder(x, self.encoder_pos_embed)

        mu = x[:, 0: self.num_latents]
        logvar = x[:, self.num_latents: 2 * self.num_latents]

        posterior = DiagonalGaussianDistribution([mu, logvar])
        latent = posterior.sample()

        return latent, center_pos, posterior


class SurfaceDecoder(nn.Module):

    def __init__(self, *,
                 num_centers: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 attn_drop_prob: float = 0.0,
                 proj_drop_prob: float = 0.0,
                 drop_path_prob: float = 0.1):

        super().__init__()

        # surface decoder
        self.query_pos_embed = nn.Parameter(torch.randn((num_centers, dim), dtype=torch.float32))
        self.center_transformer_decoder = TransformerDecoder(
            dim=dim,
            memory_dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            drop_path_prob=drop_path_prob,
            pre_norm=True
        )
        self.center_pos_head = nn.Linear(dim, 3)

    def decode_center_latents(self, z):
        bs = z.shape[0]

        queries = repeat(self.query_pos_embed, "m c -> b m c", b=bs)
        center_latents = self.center_transformer_decoder(queries, memory=z)

        center_pos = self.center_pos_head(center_latents)

        return center_latents, center_pos