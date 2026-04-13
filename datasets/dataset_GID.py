import os
import torch
import nltk
import numpy as np
import pandas as pd
import random
import ast
import re

from omegaconf import OmegaConf
from torch.utils.data import Dataset
from timm.data import create_transform
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def extract_num(s):
    parts = re.findall(r'(\D+)(\d+)', 'a' + s + '0')  # 确保首尾可匹配
    return [int(p) if p.isdigit() else p for p in sum(parts, ())]


def listdir(path):
    name_list = []
    for file in os.listdir(path):
        if 'png' in file or 'tif' in file:
            name_list.append(os.path.join(path, file))
    return name_list


def collate_fn(batch):
    img = torch.stack([b['image'] for b in batch])
    raw_caption = [b['raw_caption'] for b in batch]

    cross_image = torch.stack([b['cross_image'] for b in batch]) if 'cross_image' in batch[0].keys() else None
    cross_entity = [b['cross_entity'] for b in batch] if 'cross_entity' in batch[0].keys() else None
    cross_caption = [b['cross_caption'] for b in batch] if 'cross_caption' in batch[0].keys() else None

    return {
        'image': img,
        'raw_caption': raw_caption,
        'cross_image': cross_image,
        'cross_entity': cross_entity,
        'cross_caption': cross_caption
    }


class Dataset_GID_train(Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.data.img_dir
        self.metas = pd.read_csv(cfg.data.metas_path)
        print(f'Total {len(self.metas)} samples')

        self.img_transform = create_transform(
            input_size=cfg.data.img.img_size,
            is_training=True,
            color_jitter=cfg.data.img.color_jitter if cfg.data.img.color_jitter > 0 else None,
            interpolation=cfg.data.img.interpolation,
        )

    def __len__(self):
        return len(self.metas)

    def sample_cross_image(self, curr_meta):
        pair_index = curr_meta['pairindex']
        pair_entity = curr_meta['pairentity']

        pair_index_list = ast.literal_eval(pair_index)
        pair_entity_list = ast.literal_eval(pair_entity)

        sample_index = np.random.randint(0, len(pair_index_list))

        index = pair_index_list[sample_index]
        entity = pair_entity_list[sample_index]

        pair_meta = self.metas.iloc[index]

        filename = os.path.join(self.root_dir, pair_meta['image_id'])
        img = Image.open(filename).convert('RGB')
        caption = pair_meta['caption']
        return img, caption, entity

    def __getitem__(self, idx):
        curr_meta = self.metas.iloc[idx]
        filename = curr_meta['image_id']
        raw_caption = curr_meta['caption']

        ret_info = {}

        # img
        filename = os.path.join(self.root_dir, filename)
        img = Image.open(filename).convert('RGB')
        img = self.img_transform(img)

        # cross img
        crossimg, crosscaption, crossentity = self.sample_cross_image(curr_meta)
        crossimg = self.img_transform(crossimg)

        crossentity = 'An image of ' + crossentity
        ret_info['cross_image'] = crossimg
        ret_info['cross_entity'] = crossentity
        ret_info['cross_caption'] = crosscaption

        ret_info['image'] = img
        ret_info['raw_caption'] = raw_caption
        return ret_info
