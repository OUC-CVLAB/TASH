# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import *
import torch
from functools import partial
from timm.models.vision_transformer import vit_base_patch16_224
from timm.models.vision_transformer import _load_weights
import kornia.augmentation as Kg
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F

import copy
import logging
import random

import math
from scipy import ndimage
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia
import kornia.augmentation as K
import kornia.enhance as KE
import kornia.color as kc
import cv2
from torch.nn.modules.utils import _pair
from torchvision.transforms.functional import to_tensor
logger = logging.getLogger(__name__)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class PreEnhanceFullGPU(nn.Module):
    def __init__(self,
                 use_retinex=True, retinex_p=1.0, sigma=15.0,
                 use_clahe=True, clahe_p=1.0,
                 use_white_balance=True, white_balance_p=0.5):
        super().__init__()
        self.use_retinex = use_retinex
        self.retinex_p = retinex_p
        self.sigma = sigma

        self.use_clahe = use_clahe
        self.clahe_p = clahe_p

        self.use_white_balance = use_white_balance
        self.white_balance_p = white_balance_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clahe and torch.rand(1).item() < self.clahe_p:
            x = self.apply_clahe_tensor(x)
            # x = KE.equalize(x)

        if self.use_retinex and torch.rand(1).item() < self.retinex_p:
            x = self.apply_retinex_tensor(x, self.sigma)

        if self.use_white_balance and torch.rand(1).item() < self.white_balance_p:
            x = self.white_balance_tensor(x)

        return x

    @staticmethod
    def apply_retinex_tensor(img: torch.Tensor, sigma: float = 15.0) -> torch.Tensor:
        eps = 1e-5
        blurred = kornia.filters.gaussian_blur2d(img, (51, 51), (sigma, sigma))
        log_img = torch.log(img + eps)
        log_blur = torch.log(blurred + eps)
        retinex = log_img - log_blur
        min_val = retinex.amin(dim=(1, 2, 3), keepdim=True)
        max_val = retinex.amax(dim=(1, 2, 3), keepdim=True)
        norm_retinex = (retinex - min_val) / (max_val - min_val + eps)
        return norm_retinex.clamp(0, 1)

    @staticmethod
    def apply_clahe_tensor(img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        output = []

        for i in range(b):
            img_i = img[i]  # [C, H, W]

            if c == 1:
                img_i = img_i.expand(3, -1, -1)

            # [C, H, W] → [H, W, C]
            img_np = img_i.permute(1, 2, 0).cpu().numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            # CLAHE in LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b_ = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b_))
            enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            enhanced_tensor = to_tensor(enhanced_np)
            output.append(enhanced_tensor)

        return torch.stack(output).to(img.device)


    @staticmethod
    def white_balance_tensor(img: torch.Tensor) -> torch.Tensor:
        mean_per_channel = img.mean(dim=[2, 3], keepdim=True) # [B, C, 1, 1]
        mean_gray = img.mean(dim=[1, 2, 3], keepdim=True)
        gain = mean_gray / (mean_per_channel + 1e-5)
        img = img * gain
        return img.clamp(0, 1)

class Augmentation(nn.Module):
    def __init__(self, org_size, Aw=1.0, use_pre_enhance=False):
        super(Augmentation, self).__init__()
        self.use_pre_enhance = use_pre_enhance
        self.pre = PreEnhanceFullGPU(
            use_retinex=True, retinex_p=0.2 * Aw,
            use_clahe=True, clahe_p=0.2 * Aw,
            use_white_balance=True, white_balance_p=0.3 * Aw,
        ) if use_pre_enhance else None
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        if self.use_pre_enhance:
            x = self.pre(x)
        return self.Aug(x)


model_file = \
    '/home/ouc/data1/qiaoshishi/datasets/SUN_attributes/data_256/pretrained_places_models/pytorch_model/alexnet_places365.pth.tar'


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # self.alex = models.alexnet(num_classes=365)
        # checkpoint = T.load(model_file)
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        # self.alex.load_state_dict(state_dict)

        self.F = nn.Sequential(*list(models.alexnet(pretrained=True).features))
        # self.F = nn.Sequential(*list(self.alex.features))
        self.Pool = nn.AdaptiveAvgPool2d((6, 6))
        self.C = nn.Sequential(*list(models.alexnet(pretrained=True).classifier[:-1]))
        # self.C = nn.Sequential(*list(self.alex.classifier[:-1]))

    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = T.flatten(x, 1)
        x = self.C(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = T.flatten(x, 1)
        return x

# bgr
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class ViT(nn.Module):
    def __init__(self, hash_bit, supervised_pretrain=True):
        super(ViT, self).__init__()
        self.global_pool = False

        # lbq
        if supervised_pretrain:
            model_vit = vit_base_patch16_224(pretrained=False, drop_path_rate=0.1)
            pretrain_vit = '/mnt/8TDisk1/zhenglab/lbq/sam_ViT-B_16.npz'
            _load_weights(model_vit, checkpoint_path=pretrain_vit)
            if self.global_pool:  # set model to support GAP
                model_vit.global_pool = self.global_pool
                norm_layer = partial(nn.LayerNorm, eps=1e-6)
                embed_dim = model_vit.embed_dim
                model_vit.fc_norm = norm_layer(embed_dim)
                del model_vit.norm  # remove the original norm

        self.patch_embed = model_vit.patch_embed
        self.cls_token = model_vit.cls_token
        self.pos_embed = model_vit.pos_embed
        self.pos_drop = model_vit.pos_drop
        self.blocks = model_vit.blocks
        if self.global_pool:
            self.fc_norm = model_vit.fc_norm
        else:
            self.norm = model_vit.norm

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

class DeiT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)

    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, self.pm.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)#[B,197,768]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)#[B,12,197,64]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#[B,12,197,197]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)#[B,12,197,197]

        context_layer = torch.matmul(attention_probs, value_layer)#[B,12,197,64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)#[B,197,768]
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)#[B,197,768]
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    #归一化-多头注意力-残差连接-归一化-Mlp-残差
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)#x[B,197,768],weight[B,12,197,197]
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)#升维-激活-丢弃-降维-丢弃
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]   #[B,12,196]

        _, max_inx = last_map.max(2)    #[B,12]
        return _, max_inx

# class Encoder(nn.Module):
#     def __init__(self, config):
#         super(Encoder, self).__init__()
#         self.layer = nn.ModuleList()
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config)
#             self.layer.append(copy.deepcopy(layer))
#         self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
#
#     def forward(self, hidden_states):
#         for layer in self.layer:
#             hidden_states, weights = layer(hidden_states)
#         part_encoded = self.part_norm(hidden_states)
#         return part_encoded

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = [5, 6, 7, 8, 9, 10]
        for _ in range(config.transformer["num_layers"] - 1):
            self.layer.append(Block(config))

        self.last = Block(config)
        self.part_norm = nn.LayerNorm(config.hidden_size)
        self.part_norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.fus = TokenMLPFuser(num_layers=len(self.extract_layers), hidden_dim=config.hidden_size)
        self.gumbel_tau = 1.0
        self.a_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    def select(self,hidden_states,attn_list):
        length = len(attn_list)
        last_map = attn_list[0]
        for i in range(1, length):
            last_map = torch.matmul(attn_list[i], last_map)
        score1 = last_map[:, :, 0, 1:].max(dim=1).values.unsqueeze(-1)

        cls_token = hidden_states[:, :1, :]
        patch_tokens = hidden_states[:, 1:, :]
        norm_cls = F.normalize(cls_token, dim=-1)
        norm_patch = F.normalize(patch_tokens, dim=-1)
        score2 = torch.bmm(norm_patch, norm_cls.transpose(1, 2))

        score2 = 0.5 * (score2 + 1)
        a = torch.sigmoid(self.a_logit)  # ∈ (0,1)，learnable
        raw_score = a * score1 + (1 - a) * score2
        raw_score = raw_score.squeeze(-1)  # [B,196]

        logits = torch.stack([raw_score, 1 - raw_score], dim=-1)
        mask = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)[:, :, 0].unsqueeze(-1)

        # patch_tokens: [B, N, C]
        # mask: [B, N, 1] ∈ {0, 1}
        masked_tokens = patch_tokens * mask  # [B, N, C]

        # dynamic slection & padding
        lengths = mask.squeeze(-1).sum(dim=1).long()
        max_len = lengths.max()
        B, N, C = masked_tokens.shape
        selected_tokens = []

        for b in range(B):

            tokens_b = masked_tokens[b]  # [N, C]
            score_b = mask[b].squeeze(-1)  # [N]
            # dynamic slection
            sorted_score, sorted_indices = score_b.sort(descending=True)
            k = lengths[b]
            topk_indices = sorted_indices[:k]
            selected = tokens_b[topk_indices]  # [k, C]
            # padding
            if selected.shape[0] < max_len:
                pad_len = max_len - selected.shape[0]
                pad = torch.zeros((pad_len, C), device=patch_tokens.device)
                selected = torch.cat([selected, pad], dim=0)

            selected_tokens.append(selected)
        selected_patch_tokens = torch.stack(selected_tokens, dim=0)  # [B, max_len, C]

        combined = torch.cat([cls_token, selected_patch_tokens], dim=1)
        combined = self.part_norm(combined)

        return combined


    def forward(self, hidden_states):
        hidden_list = []
        attn_list = []

        for i, layer in enumerate(self.layer):
            hidden_states, weights = layer(hidden_states)
            if i in self.extract_layers:
                hidden_list.append(hidden_states)
                attn_list.append(weights)
                if i == self.extract_layers[-1]:
                    fused_tokens = self.fus(hidden_list)
                    hidden_states = self.select(fused_tokens, attn_list)

        part_encoded, _ = self.last(hidden_states)
        part_encoded = self.part_norm1(part_encoded)
        return part_encoded

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224,  smoothing_value=0):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.smoothing_value = smoothing_value
        self.classifier = self.config.classifier
        self.transformer = Transformer(self.config, img_size)
        #lbq:1
        # self.transformer.encoder = Encoder(config, selector_type='resmlp')  # 或 'mlp'

    def forward(self, x):
        part_tokens = self.transformer(x)  # [bs, 13, 768]
        cls_token = part_tokens[:, 0, :]
        return cls_token


    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # for bname, block in self.transformer.encoder.named_children():
            #     if bname.startswith('part') == False:
            #         for uname, unit in block.named_children():
            #             unit.load_from(weights, n_block=uname)

            # 加载 transformer encoder 中的 N-1 层
            for idx, block in enumerate(self.transformer.encoder.layer):
                block.load_from(weights, n_block=idx)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

class SwinT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)

    def forward(self, x):
        x = self.pm.patch_embed(x)
        if self.pm.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pm.pos_drop(x)
        x = self.pm.layers(x)
        x = self.pm.norm(x)  # B L C
        x = self.pm.avgpool(x.transpose(1, 2))  # B C 1
        x = T.flatten(x, 1)
        return x


class TokenMLPFuser(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=768):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, token_list):
        # stack → [B, N, C * num_layers]
        fused = torch.cat(token_list, dim=-1)
        fused = self.fusion_mlp(fused)         # [B, N, C]
        return fused


