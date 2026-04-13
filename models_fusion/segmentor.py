import os
import torch
import torch.nn as nn
import sys
import json
import cv2
import numpy as np

from torchvision import transforms
from omegaconf import OmegaConf

from models_fusion.open_clip import create_model, tokenizer
from datasets.template import openai_imagenet_template
from models_train.group_vit import GroupViT
from models_fusion.pamr import PAMR


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


class Segmentation(nn.Module):
    def __init__(self, cfg, query_words, model_path, device=torch.device('cuda'), prob_thd=0.45, logit_scale=40,
                 beta=1.2, gamma=3.0, slide_stride=128, slide_crop=256):
        super().__init__()

        self.clip = create_model('ViT-B/16', pretrained='openai', precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.groupvit = GroupViT(cfg)
        self.load_groupvit_checkpoint(model_path)
        self.groupvit = self.groupvit.half()
        for p in self.groupvit.parameters():
            p.requires_grad = False
        self.groupvit.eval().to(device)

        # self.pamr = PAMR(10, (6, 14)).to(device)
        self.pamr = PAMR(10, (2, 4)).to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.query_words, self.query_idx = self.get_cls_idx(query_words)
        self.num_queries = len(self.query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in self.query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
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

    def load_groupvit_checkpoint(self, coarse_model_path):
        print('load group-level contrastive learning model')
        pretrained_dict = torch.load(coarse_model_path)

        model_dict = self.groupvit.state_dict()

        print(model_dict.keys())
        print(pretrained_dict.keys())
        pretrained_dict = {k[12:]: v for k, v in pretrained_dict.items()
                           if k[12:] in model_dict.keys() and 'img_encoder' in k}
        print(pretrained_dict.keys())

        model_dict.update(pretrained_dict)
        self.groupvit.load_state_dict(model_dict)

    def get_cls_idx(self, query_words):
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
        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[
            1]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        out_groupvit = self.groupvit(imgs_norm)
        ex_feats = out_groupvit['attn_dict']['k'].squeeze(1).permute(0, 2, 1)
        ex_feats = ex_feats.reshape(ex_feats.shape[0], ex_feats.shape[1], 16, 16)

        image_features = self.clip.encode_image(img.half(),
                                                external_feats=ex_feats,
                                                beta=self.beta,
                                                gamma=self.gamma)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], 16, 16)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
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
            seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)

        out = self.postprocess_result(seg_logits)
        return out
