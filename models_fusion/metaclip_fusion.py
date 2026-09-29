"""Hugging Face MetaCLIP 2 dense fusion with local value preservation."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_ID = 'facebook/metaclip-2-worldwide-huge-quickgelu'


class _FusionInputReady(Exception):
    """Stop the vision forward immediately after the final layer_norm1."""


class MetaCLIPFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.config.model_type != 'metaclip_2':
            raise ValueError('Fusion requires a Hugging Face MetaCLIP 2 model.')
        self.model = model
        self.patch_size = model.config.vision_config.patch_size

    @classmethod
    def from_pretrained(cls, model_id=MODEL_ID, device='cuda', dtype=torch.float16):
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation='sdpa')
        return cls(model).to(device).eval().requires_grad_(False)

    def load_tokenizer(self, model_id=MODEL_ID):
        return AutoTokenizer.from_pretrained(model_id)

    def encode_text(self, tokens):
        # get_text_features returns a Tensor in some Transformers versions
        # and a ModelOutput in others. Use the text backbone explicitly so
        # pooling and projection are unambiguous and projection runs once.
        outputs = self.model.text_model(**tokens, return_dict=True)
        return self.model.text_projection(outputs.pooler_output)

    @staticmethod
    def custom_attn(attn_layer, x, ex_feats, beta=1.2, gamma=3.0,
                    token_size=(16, 16), local_v_weight=0.3):
        """x is final layer_norm1 output [B, 1+patches, D], including CLS.

        Keep the original similarity rule: beta was accepted but unused.
        HF v_proj is the V slice of the former packed QKV projection.
        local_v_weight mixes same-position V into the attention output:
        0 reproduces the original fusion; 1 uses only same-position V.
        """
        if not 0.0 <= local_v_weight <= 1.0:
            raise ValueError('local_v_weight must be between 0 and 1.')
        batch, length, width = x.shape
        heads = attn_layer.num_heads
        head_dim = width // heads
        if length != 1 + token_size[0] * token_size[1]:
            raise ValueError('Vision token count does not match token_size.')
        if ex_feats.ndim != 4 or ex_feats.shape[0] != batch:
            raise ValueError('ex_feats must have shape [B, C, H, W].')
        height, grid_width = ex_feats.shape[-2:]

        # Compute similarities in float32 to avoid half-precision NaNs.
        q_k = F.normalize(ex_feats.float().flatten(2), dim=1)
        similarity = q_k.transpose(1, 2) @ q_k
        similarity = (similarity - similarity.mean()) * gamma
        mask = similarity.masked_fill(similarity < 0, float('-inf'))
        # Degenerate inputs can otherwise produce a fully masked row.
        empty_rows = torch.isneginf(mask).all(dim=-1, keepdim=True)
        mask = torch.where(empty_rows, torch.zeros_like(mask), mask)

        v = attn_layer.v_proj(x)[:, 1:]
        v = v.reshape(batch, *token_size, heads, head_dim)
        v = v.permute(0, 3, 4, 1, 2).reshape(batch * heads, head_dim, *token_size)
        v = F.interpolate(v, size=(height, grid_width), mode='bilinear', align_corners=False)
        v = v.reshape(batch, heads, head_dim, height * grid_width).transpose(-1, -2)
        weights = mask.softmax(dim=-1).to(v.dtype).unsqueeze(1)
        fused = weights @ v
        # V is already aligned to the GroupViT grid. Preserve its local
        # contribution before merging heads and applying out_proj.
        if local_v_weight == 1.0:
            fused = v
        elif local_v_weight > 0.0:
            fused = local_v_weight * v + (1.0 - local_v_weight) * fused
        fused = fused.transpose(1, 2).reshape(batch, height * grid_width, width)
        return attn_layer.out_proj(fused)

    def encode_image(self, image, external_feats, beta=1.2, gamma=3.0,
                     local_v_weight=0.3):
        vision = self.model.vision_model
        last_block = vision.encoder.layers[-1]
        image = image.to(device=last_block.self_attn.v_proj.weight.device,
                         dtype=last_block.self_attn.v_proj.weight.dtype)
        # Resize the full field of view to a multiple of 14. GroupViT keeps
        # its own 16-pixel grid; value tokens are interpolated onto that grid.
        height, width = image.shape[-2:]
        size = tuple((s + self.patch_size - 1) // self.patch_size * self.patch_size
                     for s in (height, width))
        if size != (height, width):
            image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        token_size = tuple(s // self.patch_size for s in size)
        captured = []

        def capture_normalized_input(module, args, output):
            captured.append(output)
            # The original fusion skips final attention, residuals and MLP.
            raise _FusionInputReady

        handle = last_block.layer_norm1.register_forward_hook(capture_normalized_input)
        try:
            try:
                vision(pixel_values=image, interpolate_pos_encoding=True)
            except _FusionInputReady:
                pass
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Failed to capture the final MetaCLIP layer_norm1 output.')
        fused = self.custom_attn(last_block.self_attn, captured[0],
                                 ex_feats=external_feats, beta=beta, gamma=gamma,
                                 token_size=token_size, local_v_weight=local_v_weight)
        return self.model.visual_projection(vision.post_layernorm(fused))
