import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import clip

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from omegaconf import OmegaConf


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W

    ##### problems might occur here######,
    ##### N = pos.embed.shape[0] and num_patches could be N - 1
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)
    N = pos_embed.shape[1]
    # N = pos_embed.shape[1]
    if num_patches == N and W == H:
        return pos_embed

    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(H, W),
        mode='bicubic',
        align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Attention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AssignAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., hard=True,
                 gumbel=False, gumbel_tau=1., sum_assign=False, assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)
        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False, mask=None):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn, 'rawk': key, 'rawq': query, 'k': k, 'q': q}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class AttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, num_group_token, num_output_group, norm_layer=nn.LayerNorm,
                 mlp_ratio=(0.5, 4.0), hard=True, gumbel=True, sum_assign=False, assign_eps=1.,
                 gumbel_tau=1., attn_drop=0.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps,
            attn_drop=attn_drop,
        )
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)

        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)

        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))
        return new_x, attn_dict


class GroupingLayer(nn.Module):
    def __init__(self, dim, num_input_token, depth, num_heads, num_group_token, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.input_resolution = num_input_token

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x):
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
        else:
            group_token = None

        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)

        for blk_idx, blk in enumerate(self.blocks):
            cat_x = blk(cat_x)

        x, group_token = self.split_x(cat_x)

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=True)

        return x, group_token, attn_dict


class GroupViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.data.img.img_size
        self.depths = cfg.model.group_vit.depths
        self.num_layers = len(self.depths)
        self.num_heads = cfg.model.group_vit.num_heads
        self.embed_dim = cfg.model.group_vit.embed_dim
        self.patch_norm = cfg.model.group_vit.patch_norm
        self.embed_factors = cfg.model.group_vit.embed_factors
        self.num_features = int(self.embed_dim * self.embed_factors[len(self.depths) - 1])
        self.drop_rate = cfg.model.group_vit.drop_rate
        self.drop_path_rate = cfg.model.group_vit.drop_path_rate
        self.num_group_tokens = cfg.model.group_vit.num_group_tokens
        self.num_output_groups = cfg.model.group_vit.num_output_groups
        patch_size = 16

        self.patch_embed = PatchEmbed(img_size=self.img_size, kernel_size=patch_size, stride=patch_size, padding=0,
                                      in_chans=3, embed_dim=self.embed_dim, norm_layer=None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed = self.build_simple_position_embedding()
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        num_input_token = num_patches
        num_output_token = num_input_token

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(self.embed_dim * self.embed_factors[i_layer])
            downsample = None

            if i_layer < self.num_layers - 1:
                out_dim = self.embed_dim * self.embed_factors[i_layer + 1]
                downsample = GroupingBlock(dim=dim, out_dim=out_dim, num_heads=self.num_heads[i_layer],
                                           num_group_token=self.num_group_tokens[i_layer],
                                           num_output_group=self.num_output_groups[i_layer])
                num_output_token = self.num_output_groups[i_layer]

            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                num_group_token=self.num_group_tokens[i_layer],
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=downsample)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token

        self.norm = nn.LayerNorm(self.num_features)

        self.apply(self._init_weights)
        ### add cls_token to enable params loading ###
        self.pos_embed = self.build_simple_position_embedding_with_cls_token()
        self.init_backbone_with_imagenet_weights(cfg.model.dino_path)
        ### drop cls_token ###
        self.pos_embed = nn.Parameter(self.pos_embed[0, 1:])

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone_with_imagenet_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        print('Initializing ImageNet-pretrained weights')
        print('$' * 100)

        newdict = {}
        for kk, vv in state_dict.items():
            newkey = kk
            if kk.startswith('blocks.'):
                layerid = int(kk.split('.')[1])
                if 0 <= layerid < 6:
                    newkey = 'layers.0.' + kk
                elif 6 <= layerid < 12:
                    old_prefix = 'blocks.' + str(layerid) + '.'
                    new_prefix = 'blocks.' + str(layerid - 6) + '.'
                    suffix = kk.split(old_prefix)[1]
                    newkey = 'layers.1.' + new_prefix + suffix
            newdict[newkey] = vv

        del newdict['pos_embed']
        msg = self.load_state_dict(newdict, strict=False)
        print(msg)
        print('$' * 100)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_simple_position_embedding_with_cls_token(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def forward_image_head(self, x):

        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        outs = {}
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        out_attn_dict = None
        for i, layer in enumerate(self.layers):
            x, group_token, attn_dict = layer(x)
            if attn_dict is not None:
                out_attn_dict = attn_dict

        x = self.norm(x)

        outs['x_feat'] = x
        outs['x_avg'] = self.forward_image_head(x)
        outs['attn_dict'] = out_attn_dict
        return outs

