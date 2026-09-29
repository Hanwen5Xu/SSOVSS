import os

import torch
import torch.nn as nn
import sys
import json
import cv2
import numpy as np

from torchvision import transforms
from omegaconf import OmegaConf

from models_fusion.metaclip_fusion import MetaCLIPFusion, MODEL_ID
from datasets.template import openai_imagenet_template
from utils.utils import UnNormalize
from models_fusion.group_vit import GroupViT
from models_fusion.pamr import PAMR


class Segmentation(nn.Module):
    def __init__(self, cfg, name_path, device=torch.device('cuda'), prob_thd=0.3, logit_scale=40, beta=1.2, gamma=3.0,
                 slide_stride=128, slide_crop=512):
        super().__init__()
        self.cfg = cfg

        self.local_v_weight = float(OmegaConf.select(
            cfg, 'model.metaclip.local_v_weight', default=0.3))
        if not 0.0 <= self.local_v_weight <= 1.0:
            raise ValueError('model.metaclip.local_v_weight must be between 0 and 1.')

        device = torch.device(device)
        model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        model_id = OmegaConf.select(cfg, 'model.metaclip.model_id', default=MODEL_ID)
        self.metaclip = MetaCLIPFusion.from_pretrained(model_id, device, model_dtype)
        self.tokenizer = self.metaclip.load_tokenizer(model_id)

        self.groupvit = GroupViT(cfg)
        self.load_groupvit_checkpoint()
        self.groupvit = self.groupvit.to(dtype=model_dtype)
        for p in self.groupvit.parameters():
            p.requires_grad = False
        self.groupvit.eval().to(device)

        # self.pamr = PAMR(10, (6, 14)).to(device)
        self.pamr = PAMR(10, (2, 4)).to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.query_words, self.query_idx = self.get_cls_idx(name_path)
        self.num_queries = len(self.query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in self.query_words:
                query = self.tokenizer(
                    [temp(qw) for temp in openai_imagenet_template],
                    padding='max_length', truncation=True,
                    max_length=self.metaclip.model.config.text_config.max_position_embeddings,
                    return_tensors='pt').to(device)
                feature = self.metaclip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

    def load_groupvit_checkpoint(self):
        print('load groupvit pretrained')
        pretrained_dict = torch.load(self.cfg.model.group_vit.checkpoint)

        model_dict = self.groupvit.state_dict()

        print(model_dict.keys())
        print(pretrained_dict.keys())
        pretrained_dict = {k[12:]: v for k, v in pretrained_dict.items()
                           if k[12:] in model_dict.keys() and 'img_encoder' in k}
        print(pretrained_dict.keys())

        model_dict.update(pretrained_dict)
        self.groupvit.load_state_dict(model_dict)

    def get_cls_idx(self, name_path):
        with open(name_path, 'r') as file:
            query_words = json.load(file)
        print(query_words)
        query_idx = [i for i in range(len(query_words))]
        return query_words, query_idx

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    @torch.no_grad()
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        original_size = img.shape[-2:]
        pad = self.compute_padsize(*original_size, 16)
        if any(pad):
            img = nn.functional.pad(img, pad)

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.to(dtype=self.dtype)

        out_groupvit = self.groupvit(imgs_norm)
        ex_feats = out_groupvit['attn_dict']['k'].squeeze(1).permute(0, 2, 1)
        group_grid = out_groupvit['hw_shape']
        ex_feats = ex_feats.reshape(ex_feats.shape[0], ex_feats.shape[1], *group_grid)

        image_features = self.metaclip.encode_image(img.to(dtype=self.dtype),
                                                    external_feats=ex_feats,
                                                    beta=self.beta,
                                                    gamma=self.gamma,
                                                    local_v_weight=self.local_v_weight)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(img.shape[0], self.num_queries, *group_grid)

        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=False)
        if any(pad):
            l, _, t, _ = pad
            logits = logits[:, :, t:t + original_size[0], l:l + original_size[1]]
        if logit_size is not None:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 16)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding

                crop_seg_logit = self.forward_feature(crop_img).detach()

                torch.cuda.empty_cache()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')
        return logits

    def postprocess_result(self, seg_logits, data_samples=None):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)

            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred += 1  # add background
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0  # 改动

            if data_samples is None:
                return seg_pred
            # else:
            #     data_samples[i].set_data({
            #         'seg_logits':
            #             PixelData(**{'data': seg_logits}),
            #         'pred_sem_seg':
            #             PixelData(**{'data': seg_pred})
            #     })
        return data_samples

    def forward(self, inputs):
        batch_img_metas = [dict(
            ori_shape=inputs.shape[2:],
            img_shape=inputs.shape[2:],
            pad_shape=inputs.shape[2:],
            padding_size=[0, 0, 0, 0])] * inputs.shape[0]

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        if self.pamr:
            img = nn.functional.interpolate(inputs, size=inputs.shape[2:], mode='bilinear')
            img = img.cpu()
            seg_logits = seg_logits.cpu()
            self.pamr = self.pamr.cpu()

            seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)

        out = self.postprocess_result(seg_logits)
        return out
