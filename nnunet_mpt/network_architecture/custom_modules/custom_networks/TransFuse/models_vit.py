# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
import torch.nn.functional as F

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from nnunet_mpt.network_architecture.custom_modules.custom_networks.TransFuse.vision_transformer import VisionTransformer as ViT


# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
class VisionTransformer(ViT):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  # torch.Size([32, 196, 384])
        pe = self.pos_embed  # torch.Size([1, 196, 384])
        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


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


def vit_base_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        ############################################################################################
        checkpoint = torch.load('checkpoint-399_vit_l_352.pth')
        # checkpoint = torch.load('checkpoint-399_vit_l_352_491.pth')
        # checkpoint = torch.load('mae_pretrain_vit_large.pth')
        # checkpoint = torch.load('checkpoint-399_vit_l_352_kvasir.pth')
        # checkpoint = torch.load('checkpoint-399_vit_l_352_cal_polyp.pth')

        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        print(msg)
        ############################################################################################

    pe = model.pos_embed[:, 1:, :].detach()  # torch.Size([1, 196, 1024])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 1024, 196])
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])),
                 int(np.sqrt(pe.shape[2])))  # torch.Size([1, 1024, 14, 14])
    #################################### # size = trainsize / patch_size ################################
    # pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)
    pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True)

    ###################################################################################################
    pe = pe.flatten(2)  # torch.Size([1, 1024, 196])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 196, 1024])
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


def vit_huge_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if pretrained:

        ############################################################################################
        checkpoint = torch.load('checkpoint-399h.pth')
        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        print(msg)
        ############################################################################################


    pe = model.pos_embed[:, 1:, :].detach()  # torch.Size([1, 256, 1280])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 1280, 256])
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])),
                 int(np.sqrt(pe.shape[2])))  # torch.Size([1, 1280, 16, 16])
    #################################### # size = trainsize / patch_size ################################
    pe = F.interpolate(pe, size=(16, 16), mode='bilinear', align_corners=True)
    ###################################################################################################
    pe = pe.flatten(2)  # torch.Size([1, 1280, 256])
    pe = pe.transpose(-1, -2)  # torch.Size([1, 256, 1280])
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model