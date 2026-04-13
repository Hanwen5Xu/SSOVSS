import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import cv2
import numpy as np
from omegaconf import OmegaConf
import torch

from PIL import Image
import torchvision.transforms as transforms

from models_fusion.segmentor import Segmentation

cfg = OmegaConf.load(r'configs/config.yaml')

query_words = ["building", "farmland", "forest", "meadow", "water"]

net = Segmentation(cfg=cfg, query_words=query_words, model_path='checkpoint/net.pt', prob_thd=0.45).cuda()

COLORMAP = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]
CM = np.array(COLORMAP).astype('uint8')

transform_img = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

img = Image.open('imgs/32_img.png')
img = transform_img(img).cuda()
img = img.unsqueeze(0)

out = net(img)

out = out.cpu().data.numpy()
out_color = CM[out][0]
out_color = cv2.cvtColor(out_color, cv2.COLOR_BGR2RGB)
cv2.imwrite(f'visualization/out.png', out_color)
