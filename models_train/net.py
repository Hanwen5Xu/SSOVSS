import os

import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import diffdist.functional as diff_dist

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import AutoModel
from einops import rearrange, repeat
from omegaconf import OmegaConf

from models_train.group_vit import GroupViT
from models_train.losses import HungarianMatcher, dice_loss


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


def resize_attn_map(attentions, h, w, align_corners=False):
    scale = (h * w // attentions.shape[2]) ** 0.5
    if h > w:
        w_featmap = w // int(np.round(scale))
        h_featmap = attentions.shape[2] // w_featmap
    else:
        h_featmap = h // int(np.round(scale))
        w_featmap = attentions.shape[2] // h_featmap
    assert attentions.shape[
               2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

    bs = attentions.shape[0]
    nh = attentions.shape[1]  # number of head
    groups = attentions.shape[3]  # number of group token
    # [bs, nh, h*w, groups] -> [bs*nh, groups, h, w]
    attentions = rearrange(
        attentions, 'bs nh (h w) c -> (bs nh) c h w', bs=bs, nh=nh, h=h_featmap, w=w_featmap, c=groups)
    attentions = F.interpolate(attentions, size=(h, w), mode='bilinear', align_corners=align_corners)
    #  [bs*nh, groups, h, w] -> [bs, nh, h*w, groups]
    attentions = rearrange(attentions, '(bs nh) c h w -> bs nh h w c', bs=bs, nh=nh, h=h, w=w, c=groups)

    return attentions


class ProjectMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:
        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


class DistilBert(nn.Module):
    def __init__(
            self,
            context_length: int,
            width: int,
            layers: int,
            vocab_size,
            use_checkpoint=False,
            pretrained=True,
            fixed=True,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased',
                                                     output_hidden_states=True)
        self.transformer.train()
        self.width = width

        if fixed is True:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if pretrained is False:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, as_dict=True):
        outs = {}
        out_x = self.transformer(**x)

        out_hidden = out_x.last_hidden_state[:, 0, :]
        last_hidden = out_x.hidden_states[-1]

        outs['x'] = out_hidden
        outs['all_tokens'] = last_hidden
        return outs


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_encoder = GroupViT(cfg)
        self.text_encoder = DistilBert(context_length=cfg.model.text_encoder.context_length,
                                       width=cfg.model.text_encoder.width,
                                       layers=cfg.model.text_encoder.layers,
                                       vocab_size=cfg.model.text_encoder.vocab_size,
                                       pretrained=cfg.model.text_encoder.pretrained,
                                       fixed=cfg.model.text_encoder.fixed)

        self.contrast_temperature = cfg.model.contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.contrast_temperature))

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)

        self.img_projector = ProjectMLP(in_dim=768, num_layers=2, out_dim=256)
        self.text_projector = ProjectMLP(in_dim=768, num_layers=2, out_dim=256)
        self.img_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_projector)
        self.text_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_projector)

        ## for mask loss ###
        self.maskloss_weight = 1
        self.cross_threshold = 0.6
        self.topmask_ratio = 1.0
        self.dual_dice = False
        self.group_ratio = 0.5
        self.num_deep_stages = 1
        self.logit_scale_mask = nn.Parameter(torch.ones([]) * np.log(1 / cfg.model.contrast_temperature))

        self.img_encoder_momentum = GroupViT(cfg)

        self.q_projector = nn.Identity()
        self.k_projector = nn.Identity()
        self.q_projector_momentum = nn.Identity()
        self.k_projector_momentum = nn.Identity()
        ## set momentum branch offline
        for param_q, param_k in zip(self.img_encoder.parameters(), self.img_encoder_momentum.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.student_temp = 1
        self.teacher_temp = 0.07
        self.teacher_momentum = 0.99

        self.K = int(cfg['train']['num_instances'] / cfg['train']['batch_size'] *
                     cfg['train']['epochs'])
        self.k = int(cfg['train']['num_instances'] / cfg['train']['batch_size'] * (
            cfg['train']['start_epoch']))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.img_encoder.parameters(), self.img_encoder_momentum.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def encode_image(self, image, momentum=False):
        outs = {}
        if momentum:
            with torch.no_grad():
                img_outs = self.img_encoder_momentum(image)

                outs['image_x_before_proj'] = img_outs['x_avg']
                outs['image_x'] = self.img_projector(img_outs['x_avg'])
                outs['image_feat_before_proj'] = img_outs['x_feat']
                outs['image_feat'] = self.img_projector(img_outs['x_feat'])
                outs['attn_dict'] = img_outs['attn_dict']
        else:
            img_outs = self.img_encoder(image)

            outs['image_x_before_proj'] = img_outs['x_avg']
            outs['image_x'] = self.img_projector(img_outs['x_avg'])
            outs['image_feat_before_proj'] = img_outs['x_feat']
            outs['image_feat'] = self.img_projector(img_outs['x_feat'])
            outs['attn_dict'] = img_outs['attn_dict']
        return outs

    def encode_text(self, text):
        outs = {}
        text_outs = self.text_encoder(text)
        text_x = text_outs['x']
        text_all_tokens = text_outs['all_tokens']

        outs['text_x_before_proj'] = text_x
        outs['text_x'] = self.text_projector(text_x)
        outs['text_feat_before_proj'] = text_all_tokens
        outs['text_feat'] = self.text_projector(text_all_tokens)

        return outs

    def matching_loss(self, image_x, text_x):
        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        # labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device)

        image_x = F.normalize(image_x, dim=-1)  # [B, C]
        text_x = F.normalize(text_x, dim=-1)  # [B, C]

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        # logits_per_img = image_x @ text_x.t()
        # logits_per_text = text_x @ image_x.t()
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)
        return loss

    def project_and_mask(self, q, k, branch='online'):
        scale = self.img_encoder.width ** -0.5

        if branch == 'online':
            q = self.q_projector(q)
            k = self.k_projector(k)
            attn = q @ k.transpose(-2, -1) * scale  ### no softmax for now
        else:
            with torch.no_grad():
                q = self.q_projector_momentum(q)
                k = self.k_projector_momentum(k)
                attn = q @ k.transpose(-2, -1) * scale  ### no softmax for now

        return attn

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        # k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        k = F.softmax(k / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def compute_cross_loss(self, mask_cross1, mask2, cross_entity, groups1, groups2, indicator='none'):
        bs = mask_cross1.shape[0]
        mask_cross1 = rearrange(mask_cross1, 'b k (h w) -> b k h w', h=16, w=16)  # hard coded this for now, [b, h, w]
        mask2 = rearrange(mask2, 'b k (h w) -> b k h w', h=16, w=16)  # hard coded this for now, [b, h, w]

        with torch.no_grad():
            noun_feat = self.encode_text(cross_entity)['text_x']  # [bs, d_c]
            group_logits1 = (groups1 @ noun_feat.unsqueeze(-1)).squeeze(-1)  # [bs, k]
            group_logits2 = (groups2 @ noun_feat.unsqueeze(-1)).squeeze(-1)  # [bs, k]
            topk_logits1 = torch.topk(group_logits1, k=1, largest=True)[1]
            topk_logits2 = torch.topk(group_logits2, k=1, largest=True)[1]

        mask_select1 = mask_cross1[torch.arange(bs).unsqueeze(1), topk_logits1]
        mask_select2 = mask2[torch.arange(bs).unsqueeze(1), topk_logits2]
        mask_select1 = mask_select1.flatten(1)
        mask_select2 = mask_select2.flatten(1)

        mask_loss = self.self_distill(mask_select1, mask_select2)
        return mask_loss

    def forward(self, image, text, cross_image=None, cross_entity=None):
        losses_dict = dict()

        image_outs = self.encode_image(image, momentum=False)
        text_outs = self.encode_text(text)

        image_x = image_outs['image_x']
        text_x = text_outs['text_x']

        matchingloss = self.matching_loss(image_x, text_x)
        losses_dict['matching'] = matchingloss

        # cross loss
        image_outs2 = self.encode_image(cross_image, momentum=True)
        attn_q = image_outs['attn_dict']['q'].squeeze(1)
        attn_k = image_outs['attn_dict']['k'].squeeze(1)
        attn_q_cross = image_outs2['attn_dict']['q'].squeeze(1)
        attn_k_cross = image_outs2['attn_dict']['k'].squeeze(1)

        attn_map2 = self.project_and_mask(attn_q_cross, attn_k_cross)
        attn_map_cross1 = self.project_and_mask(attn_q, attn_k_cross)

        maskloss = self.compute_cross_loss(attn_map_cross1, attn_map2, cross_entity, image_outs['image_feat'],
                                           image_outs2['image_feat'], indicator='none')

        losses_dict['mask'] = self.maskloss_weight * maskloss

        losses = matchingloss + self.maskloss_weight * maskloss
        losses_dict['loss'] = losses
        self._momentum_update_key_encoder()  # update the key encoder
        return losses_dict

    def inference(self, imgs, infer_tokens):
        add_bg = True
        bg_thresh = 0.2

        img_outs = self.encode_image(imgs)
        text_outs = self.encode_text(infer_tokens)

        # get attention maps
        attn_map = img_outs['attn_dict']['soft']  # [B, nH, G, HxW]
        attn_map = rearrange(attn_map, 'b h g n -> b h n g')  # [B, nH, HxW, G]
        attn_map = resize_attn_map(attn_map, *imgs.shape[-2:])  # [B, nH, H, W, G]
        attn_map = attn_map.squeeze(1)  # [B, H, W, G]

        # get group features
        # grouped_img_tokens = img_outs['image_feat'].squeeze(0)  # [B, G, C]
        grouped_img_tokens = img_outs['image_feat']  # [B, G, C]

        img_avg_feat = img_outs['image_x']  # [B, C]
        grouped_img_tokens = F.normalize(grouped_img_tokens, dim=-1)
        img_avg_feat = F.normalize(img_avg_feat, dim=-1)

        onehot_attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)
        num_fg_classes = text_outs['text_x'].shape[0]
        class_offset = 1 if add_bg else 0
        text_tokens = text_outs['text_x']
        text_tokens = F.normalize(text_tokens, dim=-1)
        num_classes = num_fg_classes + class_offset

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        # get affinity mat
        group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
        group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)
        avg_affinity_mat = (img_avg_feat @ text_tokens.T) * logit_scale
        avg_affinity_mat = F.softmax(avg_affinity_mat, dim=-1)
        avg_affinity_mat = avg_affinity_mat.unsqueeze(1)

        affinity_mask = torch.zeros_like(avg_affinity_mat)
        avg_affinity_topk = avg_affinity_mat.topk(dim=-1, k=min(5, num_fg_classes))
        affinity_mask.scatter_add_(
            dim=-1, index=avg_affinity_topk.indices, src=torch.ones_like(avg_affinity_topk.values))
        group_affinity_mat.masked_fill_(~affinity_mask.bool(), float('-inf'))
        group_affinity_mat = F.softmax(group_affinity_mat, dim=-1)

        # get result
        pred_logits = torch.zeros(imgs.shape[0], num_classes, *imgs.shape[-2:], device=imgs.device, dtype=imgs.dtype)
        temp = torch.einsum('bijk,bkl->bijl', onehot_attn_map, group_affinity_mat)
        pred_logits[:, class_offset:] = rearrange(temp, 'b h w c -> b c h w')

        if add_bg:
            bg_thresh = min(bg_thresh, group_affinity_mat.max().item())
            pred_logits = rearrange(pred_logits, 'b c h w -> c b h w')
            pred_logits[0, temp.max(dim=-1).values < bg_thresh] = 1
            pred_logits = rearrange(pred_logits, 'c b h w -> b c h w')
        return pred_logits
