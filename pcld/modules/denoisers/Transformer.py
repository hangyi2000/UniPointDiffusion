# https://github.com/GuyTevet/motion-diffusion-model/blob/main/model/mdm.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
# from model.rotation2xyz import Rotation2xyz
import random


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    

class PositionalEncoding_v505(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_v505, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # self.register_buffer('pe', pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return 
    

class myTransfomerModel(nn.Module):
    # def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
    #              latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
    #              ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
    #              arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        # self.legacy = legacy
        # self.modeltype = modeltype
        # self.njoints = njoints
        # self.nfeats = nfeats
        # self.num_actions = num_actions
        # self.data_rep = data_rep
        # self.dataset = dataset

        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        # self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # self.ablation = ablation
        self.activation = activation
        # self.clip_dim = clip_dim
        # self.action_emb = kargs.get('action_emb', None)

        # self.input_feats = self.njoints * self.nfeats

        # self.normalize_output = kargs.get('normalize_encoder_output', False)

        # self.cond_mode = kargs.get('cond_mode', 'no_cond')
        # self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        # self.arch = arch
        # self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        # self.emb_trans_dec = emb_trans_dec

        # if self.arch == 'trans_enc':
        #     print("TRANS_ENC init")
        #     seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                       nhead=self.num_heads,
        #                                                       dim_feedforward=self.ff_size,
        #                                                       dropout=self.dropout,
        #                                                       activation=self.activation)

        #     self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
        #                                                  num_layers=self.num_layers)
        # elif self.arch == 'trans_dec':
        #     print("TRANS_DEC init")
        #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
        #                                                       nhead=self.num_heads,
        #                                                       dim_feedforward=self.ff_size,
        #                                                       dropout=self.dropout,
        #                                                       activation=activation)
        #     self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
        #                                                  num_layers=self.num_layers)
        # elif self.arch == 'gru':
        #     print("GRU init")
        #     self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        # else:
        #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # if self.cond_mode != 'no_cond':
        #     if 'text' in self.cond_mode:
        #         self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        #         print('EMBED TEXT')
        #         print('Loading CLIP...')
        #         self.clip_version = clip_version
        #         self.clip_model = self.load_and_freeze_clip(clip_version)
        #     if 'action' in self.cond_mode:
        #         self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
        #         print('EMBED ACTION')

        # self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
        #                                     self.nfeats)

        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # bs, njoints, nfeats, nframes = x.shape
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        #     enc_text = self.encode_text(y['text'])
        #     emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        # if 'action' in self.cond_mode:
        #     action_emb = self.embed_action(y['action'])
        #     emb += self.mask_cond(action_emb, force_mask=force_mask)

        emb += y.transpose(0, 1)

        # if self.arch == 'gru':
        #     x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
        #     emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
        #     emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
        #     emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
        #     x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        # x = self.input_process(x)

        # if self.arch == 'trans_enc':
        #     # adding the timestep embed
        #     xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        #     xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        #     output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # elif self.arch == 'trans_dec':
        #     if self.emb_trans_dec:
        #         xseq = torch.cat((emb, x), axis=0)
        #     else:
        #         xseq = x
        #     xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        #     if self.emb_trans_dec:
        #         output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
        #     else:
        #         output = self.seqTransDecoder(tgt=xseq, memory=emb)
        # elif self.arch == 'gru':
        #     xseq = x
        #     xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
        #     output, _ = self.gru(xseq)

        # adding the timestep embed
        xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        output = output.transpose(0, 1)
        return output


class myTransfomerModel_v4(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 两个 emb 了，需要再往后数一位
        output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        return output
    
class myTransformerModel_class_v101(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 两个 emb 了，需要再往后数一位
        output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


class myTransformerModel_class_v103(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 两个 emb 了，需要再往后数一位
        output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output
    

class myTransformerModel_class_v003(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768, guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.guidance_uncondp > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        y = self.mask_cond(y)

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 两个 emb 了，需要再往后数一位
        output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


class myTransformerModel_class_v203(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768, guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.guidance_uncondp > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        y = self.mask_cond(y, force_mask=force_mask)

        # adding the timestep embed
        # xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        # 还是像原先 mdm 一样将 time-embedding 和 label-embedding 加在一起
        emb += y.transpose(0, 1)
        xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # # 两个 emb 了，需要再往后数一位
        # output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output
    

class myTransformerModel_img_v101(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 50+1 个 emb 了，需要再往后数51位
        output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


# class myTransformerModel_class_v101(nn.Module):
class myTransformerModel_img_v102(nn.Module):
    # 与 myTransformerModel_class_v101 保持一致
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)
        
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        # adding the timestep embed
        xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+2, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # 两个 emb 了，需要再往后数一位
        output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output
    

# class myTransformerModel_class_v203(nn.Module):
class myTransformerModel_img_v203(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768, guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.guidance_uncondp > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 不把 label-embedding 和 time-embedding 加在一起了
        # emb += y.transpose(0, 1)

        # 不把 label-embedding 和 time-embedding 加在一起了
        # # adding the timestep embed
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        y = self.mask_cond(y, force_mask=force_mask)

        # adding the timestep embed
        # xseq = torch.cat((y.transpose(0, 1), emb, x.transpose(0, 1)), axis=0)  # [seqlen+2, bs, d]
        # 还是像原先 mdm 一样将 time-embedding 和 label-embedding 加在一起
        emb += y.transpose(0, 1)
        xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # # 两个 emb 了，需要再往后数一位
        # output = self.seqTransEncoder(xseq)[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


# class myTransformerModel_class_v203(nn.Module):
class myTransformerModel_img_v204(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768, guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.guidance_uncondp > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # v204 中输入变成了一个 50(7*7+1)*token 的 embedding
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 一系列的 proj
        x = self.sample_proj(x)
        y = self.context_proj(y)

        y = self.mask_cond(y, force_mask=force_mask)

        # emb += y.transpose(0, 1)
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        xseq = torch.cat((emb, y.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output
    

class myTransformerModel_single_category_v305(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu", sample_dim=768, context_dim=768, guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        self.context_proj = nn.Linear(context_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.guidance_uncondp > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # v204 中输入变成了一个 50(7*7+1)*token 的 embedding
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        # 一系列的 proj
        x = self.sample_proj(x)
        if y is not None:
            y = self.context_proj(y)
            y = self.mask_cond(y, force_mask=force_mask)

        # emb += y.transpose(0, 1)
        # xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)  # [seqlen+1, bs, d]
        if y is not None:
            xseq = torch.cat((emb, y.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]
        else:
            xseq = torch.cat((emb, x.transpose(0, 1)), axis=0)
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        if y is not None:
            output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        else:
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


class myTransformerModel_multi_v405(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, 
                 dropout=0.1, dropout_each_cond=0.0, dropout_all_cond=0.0, retain_all_cond=0.0, 
                 activation="gelu", sample_dim=768, img_dim=768, text_dim=512, class_dim=768, 
                 guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_each_cond = dropout_each_cond
        self.dropout_all_cond = dropout_all_cond
        self.retain_all_cond = retain_all_cond

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        # self.context_proj = nn.Linear(context_dim, latent_dim)
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.class_proj = nn.Linear(class_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    # def mask_cond(self, cond, force_mask=False):
    #     # bs, d = cond.shape
    #     # 有三个维度，view 也要改
    #     bs = cond.shape[0]
    #     if force_mask:
    #         return torch.zeros_like(cond)
    #     elif self.training and self.guidance_uncondp > 0.:
    #         # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
    #         mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
    #         return cond * (1. - mask)
    #     else:
    #         return cond

    def mask_cond(self, cond, mask_p=0.0, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and mask_p > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * mask_p).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        class_num_emb, img_emb, text_emb = y

        # 一系列的 proj
        x = self.sample_proj(x)
        # y = self.context_proj(y)
        class_cond = self.class_proj(class_num_emb)
        img_cond = self.img_proj(img_emb)
        text_cond = self.text_proj(text_emb)

        # # 进行 mask 的操作
        # y = self.mask_cond(y, force_mask=force_mask)
        # 每个模态分别进行 self.dropout_each_cond 的随机 dropout
        # 所有模态一起进行 self.dropout_all_cond 的随机 dropout
        # 所有模态一起进行 self.retain_all_cond 的随机 retain
        dropout_each_cond_thread = 1 - (self.dropout_all_cond + self.retain_all_cond)   # 随机数小于 0.8 进行随机 mask
        dropout_all_cond_thread = dropout_each_cond_thread + self.dropout_all_cond    # 随机数 [0.8, 0.9] 维持所有 cond 不动

        rnd_num = random.random()
        if force_mask:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        elif rnd_num < dropout_each_cond_thread:
            class_cond = self.mask_cond(class_cond, mask_p=self.dropout_each_cond)
            img_cond = self.mask_cond(img_cond, mask_p=self.dropout_each_cond)
            text_cond = self.mask_cond(text_cond, mask_p=self.dropout_each_cond)
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
        elif rnd_num < dropout_all_cond_thread:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        else:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)

        # xseq = torch.cat((emb, y.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]
        xseq = torch.cat((emb, cond.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.seqTransEncoder(xseq)[-1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output


class myTransformerModel_multi_v405_ablation(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, 
                 dropout=0.1, dropout_each_cond=0.0, dropout_all_cond=0.0, retain_all_cond=0.0, 
                 activation="gelu", sample_dim=768, img_dim=768, text_dim=512, class_dim=768, 
                 guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_each_cond = dropout_each_cond
        self.dropout_all_cond = dropout_all_cond
        self.retain_all_cond = retain_all_cond

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        self.sample_proj = nn.Linear(sample_dim, latent_dim)
        # condition projector
        # self.context_proj = nn.Linear(context_dim, latent_dim)
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.class_proj = nn.Linear(class_dim, latent_dim)
        # output
        self.output_proj = nn.Linear(latent_dim, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    # def mask_cond(self, cond, force_mask=False):
    #     # bs, d = cond.shape
    #     # 有三个维度，view 也要改
    #     bs = cond.shape[0]
    #     if force_mask:
    #         return torch.zeros_like(cond)
    #     elif self.training and self.guidance_uncondp > 0.:
    #         # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
    #         mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
    #         return cond * (1. - mask)
    #     else:
    #         return cond

    def mask_cond(self, cond, mask_p=0.0, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and mask_p > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * mask_p).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        class_num_emb, img_emb, text_emb = y

        # 一系列的 proj
        x = self.sample_proj(x)
        # y = self.context_proj(y)
        class_cond = self.class_proj(class_num_emb)
        img_cond = self.img_proj(img_emb)
        text_cond = self.text_proj(text_emb)

        # # 进行 mask 的操作
        # y = self.mask_cond(y, force_mask=force_mask)
        # 每个模态分别进行 self.dropout_each_cond 的随机 dropout
        # 所有模态一起进行 self.dropout_all_cond 的随机 dropout
        # 所有模态一起进行 self.retain_all_cond 的随机 retain
        dropout_each_cond_thread = 1 - (self.dropout_all_cond + self.retain_all_cond)   # 随机数小于 0.8 进行随机 mask
        dropout_all_cond_thread = dropout_each_cond_thread + self.dropout_all_cond    # 随机数 [0.8, 0.9] 维持所有 cond 不动

        rnd_num = random.random()
        if force_mask:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        elif rnd_num < dropout_each_cond_thread:
            class_cond = self.mask_cond(class_cond, mask_p=self.dropout_each_cond)
            img_cond = self.mask_cond(img_cond, mask_p=self.dropout_each_cond)
            text_cond = self.mask_cond(text_cond, mask_p=self.dropout_each_cond)
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
        elif rnd_num < dropout_all_cond_thread:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        else:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)

        # xseq = torch.cat((emb, y.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]
        xseq = torch.cat((emb, cond.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.seqTransEncoder(xseq)[-1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1)
        output = self.output_proj(output)
        return output
    

class myTransformerModel_multi_v505(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, 
                 dropout=0.1, dropout_each_cond=0.0, dropout_all_cond=0.0, retain_all_cond=0.0, 
                 activation="gelu", sample_dim=768, img_dim=768, text_dim=512, class_dim=768, 
                 guidance_uncondp=0.1):
        super().__init__()
        # todo：
        # 加入fc

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_each_cond = dropout_each_cond
        self.dropout_all_cond = dropout_all_cond
        self.retain_all_cond = retain_all_cond

        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding_v505(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            norm_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # input
        # self.sample_proj = nn.Linear(sample_dim, latent_dim)
        self.sample_proj_first = nn.Linear(sample_dim, latent_dim)
        self.sample_proj_second = nn.Linear(sample_dim, latent_dim)
        self.sample_proj_third = nn.Linear(sample_dim, latent_dim)
        self.sample_proj_fourth = nn.Linear(sample_dim, latent_dim)
        # condition projector
        # self.context_proj = nn.Linear(context_dim, latent_dim)
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.class_proj = nn.Linear(class_dim, latent_dim)
        # output
        # self.output_proj = nn.Linear(latent_dim, sample_dim)
        self.output_proj = nn.Linear(latent_dim*4, sample_dim)

        self.guidance_uncondp = guidance_uncondp

    # def mask_cond(self, cond, force_mask=False):
    #     # bs, d = cond.shape
    #     # 有三个维度，view 也要改
    #     bs = cond.shape[0]
    #     if force_mask:
    #         return torch.zeros_like(cond)
    #     elif self.training and self.guidance_uncondp > 0.:
    #         # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
    #         mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
    #         return cond * (1. - mask)
    #     else:
    #         return cond

    def mask_cond(self, cond, mask_p=0.0, force_mask=False):
        # bs, d = cond.shape
        # 有三个维度，view 也要改
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and mask_p > 0.:
            # mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.guidance_uncondp).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * mask_p).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, y=None, force_mask=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] != bs:
            emb = emb.repeat([1, bs, 1])

        class_num_emb, img_emb, text_emb = y

        # 一系列的 proj
        # x = self.sample_proj(x)
        x_first = self.sample_proj_first(x)
        x_second = self.sample_proj_second(x)
        x_third = self.sample_proj_third(x)
        x_fourth = self.sample_proj_fourth(x)
        # y = self.context_proj(y)
        class_cond = self.class_proj(class_num_emb)
        img_cond = self.img_proj(img_emb)
        text_cond = self.text_proj(text_emb)

        # # 进行 mask 的操作
        # y = self.mask_cond(y, force_mask=force_mask)
        # 每个模态分别进行 self.dropout_each_cond 的随机 dropout
        # 所有模态一起进行 self.dropout_all_cond 的随机 dropout
        # 所有模态一起进行 self.retain_all_cond 的随机 retain
        dropout_each_cond_thread = 1 - (self.dropout_all_cond + self.retain_all_cond)   # 随机数小于 0.8 进行随机 mask
        dropout_all_cond_thread = dropout_each_cond_thread + self.dropout_all_cond    # 随机数 [0.8, 0.9] 维持所有 cond 不动

        rnd_num = random.random()
        if force_mask:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        elif rnd_num < dropout_each_cond_thread:
            class_cond = self.mask_cond(class_cond, mask_p=self.dropout_each_cond)
            img_cond = self.mask_cond(img_cond, mask_p=self.dropout_each_cond)
            text_cond = self.mask_cond(text_cond, mask_p=self.dropout_each_cond)
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
        elif rnd_num < dropout_all_cond_thread:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)
            cond = self.mask_cond(cond, force_mask=True)
        else:
            cond = torch.cat((class_cond, img_cond, text_cond), axis=1)

        # xseq = torch.cat((emb, y.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]
        # xseq = torch.cat((emb, cond.transpose(0, 1), x.transpose(0, 1)), axis=0)  # [t_emb_len+cond_emb_len+seqlen, bs, d] 也就是 [1+50+1, bs, d]
        xseq = torch.cat((emb, cond.transpose(0, 1), 
                          x_first.transpose(0, 1), 
                          x_second.transpose(0, 1), 
                          x_third.transpose(0, 1), 
                          x_fourth.transpose(0, 1)), axis=0)

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # output = self.seqTransEncoder(xseq)[51:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        # output = self.seqTransEncoder(xseq)[-1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.seqTransEncoder(xseq)[-4:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = output.transpose(0, 1) # [bs, 4, d]
        output = output.reshape(bs, 1, -1)
        output = self.output_proj(output)
        return output
    

if __name__ == "__main__":
    # import time
    # T1 = time.time()
    # model = myTransformerModel_single_category_v305(latent_dim=768)
    # print("create model completed")
    # T2 = time.time()
    # print('UNetModel初始化时间:%s毫秒' % ((T2 - T1)*1000))# 程序运行时间:0.0毫秒
    # x = torch.rand(16, 1, 768)
    # timesteps = torch.randint(0, 1000, (16,))
    # timesteps = timesteps.long()
    # context = torch.rand(16, 1, 768)
    # output = model(x, timesteps, context)
    # print(output.shape)
    # # print(output)
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    import time
    T1 = time.time()
    model = myTransformerModel_multi_v405(latent_dim=768)
    print("create model completed")
    T2 = time.time()
    print('UNetModel初始化时间:%s毫秒' % ((T2 - T1)*1000))# 程序运行时间:0.0毫秒

    x = torch.rand(16, 1, 768)
    timesteps = torch.randint(0, 1000, (16,))
    timesteps = timesteps.long()

    class_emb = torch.rand(16, 1, 768)
    img_emb = torch.rand(16, 50, 768)
    text_emb = torch.rand(16, 5, 512)
    output = model(x, timesteps, (class_emb, img_emb, text_emb))
    print(output.shape)
    # print(output)

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))