import argparse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import warnings
import os
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import kornia.augmentation as Kg
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from collections import OrderedDict

from easydict import EasyDict as edict


config = edict()

config.patches = edict()
config.patches.size = (16, 16)

config.split = 'non-overlap'
config.slide_step = 12
config.hidden_size = 768

config.transformer = edict()
config.transformer.mlp_dim = 3072
config.transformer.num_heads = 12
config.transformer.num_layers = 12
config.transformer.attention_dropout_rate = 0.0
config.transformer.dropout_rate = 0.1

config.classifier = 'token'
config.representation_size = None












# 用于将 args 参数复制进 config（在 main_DHD.py 中调用）
def update_from_args(args):
    config.update(vars(args))

# from config import config, update_from_args
# update_from_args(args)
# print(config.batch_size)  # 与 args.batch_size 等价
