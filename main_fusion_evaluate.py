import os
import cv2
import re
import numpy as np
import torch
import random
from omegaconf import OmegaConf

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.utils as vutils

from models_fusion.segmentor import Segmentation
from datasets.dataset_GID import DataSet_GID_Val
from utils.metrics import Evaluator


cfg = OmegaConf.load(r'configs/config_fusion.yaml')
dataset_val = DataSet_GID_Val(cfg)
loader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, drop_last=False, shuffle=False)

net = Segmentation(cfg=cfg, name_path='datasets/gid.json').cuda()

metric = Evaluator(num_class=6)
meanimg = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, -1, 1, 1).cuda()
stdimg = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, -1, 1, 1).cuda()

jj = 0
for img, label in loader_val:
    img = img.cuda()
    out = net(img)

    out = out.cpu().data.numpy()
    label = label.data.numpy()

    out[label == 0] = 0

    metric.add_batch(label, out)

    jj += 1
    if jj % 100 == 0:
        print(jj)

val_MIOU = metric.Mean_Intersection_over_Union()
val_PA = metric.Pixel_Accuracy()
val_F1 = metric.F1_score()
print('val_MIOU: %.4f val_PA: %.4f val_F1: %.4f'
      % (val_MIOU, val_PA, val_F1))
