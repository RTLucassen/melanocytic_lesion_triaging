# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytorch implementation of Vision Transformer (ViT)

Based on the HIPT reimplementation by Clement Grisi:
https://github.com/clemsgrs/hipt

Which was in part copied from the original HIPT implementation by Chen et al.:
https://github.com/mahmoodlab/HIPT

Which was in part copied from the original DINO implementation by Caron et al.:
https://github.com/facebookresearch/dino

Which was in part copied from the timm library by Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import warnings


class LearnablePosEmbedding(nn.Module):
    # indicates whether the interpolation of the positional embeddings 
    # (if necessary) is implemented in the same way as the original 
    # implementation used in HIPT
    match_implementation = True

    def __init__(
        self, 
        embed_dim: int, 
        expected_shape: tuple[int, int], 
        dropout_prob: float,
    ) -> None:
        """
        Initialize class for generating learnable positional embeddings.

        Args:
            embed_dim:  Dimensionality of embeddings.
            expected_shape:  Expected spatial shape of embeddings before sequence.
            dropout_prob:  Probablity of dropout.
        """
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.expected_shape = expected_shape

        # initialize learnable positional embedding        
        sequence_length = self.expected_shape[0]*self.expected_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length+1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        
        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)

    def forward(
        self, 
        x: torch.Tensor, 
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Add positional embedding to sequence of embeddings.

        Args:
            x:  Batch of embeddings as (batch, sequence, channels).
            shape:  Actual spatial shape of the input as (height, width).
        
        Returns:
            x:  Batch of embeddings as (batch, sequence, channels).
        """
        # interpolate the positional embeddings if necessary
        if not isinstance(shape, tuple):
            raise ValueError('Invalid argument for shape.')
        elif len(shape) != 2:
            raise ValueError('Invalid length of argument for shape.')
        elif shape != self.expected_shape:
            # separate the cls positional embedding from the rest of the sequence
            cls_pos_embed = self.pos_embed[:, 0:1]
            seq_pos_embed = self.pos_embed[:, 1:]
            # reshape the sequence of positional embeddings
            spatial_shape = (1, *self.expected_shape, self.embed_dim)
            spatial_pos_embed = seq_pos_embed.reshape(spatial_shape)
            spatial_pos_embed = spatial_pos_embed.permute(0, 3, 1, 2)
            # interpolate the positional embeddings
            if self.match_implementation:
                # we add a small number (0.1) to avoid floating point error in 
                # the interpolation. See the discussion at:
                # https://github.com/facebookresearch/dino/issues/8
                spatial_pos_embed = nn.functional.interpolate(
                    input=spatial_pos_embed,
                    scale_factor=((shape[0]+0.1) / self.expected_shape[0],
                                (shape[1]+0.1) / self.expected_shape[1]),
                    mode='bicubic',
                )
            else:
                spatial_pos_embed = nn.functional.interpolate(
                    input=spatial_pos_embed,
                    size=shape,
                    mode='bicubic',
                )
            # reshape to the sequence of positional embeddings
            spatial_pos_embed = spatial_pos_embed.permute(0, 2, 3, 1)
            seq_pos_embed = spatial_pos_embed.reshape(1, -1, self.embed_dim)
            # combine the cls positional embedding with the rest of the sequence
            pos_embed = torch.cat((cls_pos_embed, seq_pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed
        # add positional embedding to each element in the sequence
        x = self.pos_drop(x+pos_embed)
        
        return x


class PosEmbedding(nn.Module):

    max_position_index = 100
    repeat_section_embedding = True

    def __init__(
        self, 
        embed_dim: int, 
        dropout_prob: float,
        learnable_weight: bool = False,
    ) -> None:
        """
        Initialize class for generating predefined positional embeddings.

        Args:
            embed_dim:  Dimensionality of embeddings.
            dropout_prob:  Probablity of dropout.
            learnable_weight:  Indicates whether the positional embedding and
                section embedding are first multiplied with a learnable weight
                before being added to the feature vectors.
        """
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.learnable_weight = learnable_weight
        self.pos_embed = nn.Parameter(
            data=torch.zeros((self.max_position_index+1, self.embed_dim//2)),
            requires_grad=False,
        )
        X = torch.arange(self.max_position_index+1, dtype=torch.float32).reshape(-1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2))
        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        # initialize learnable weights if specified
        if self.learnable_weight:
            self.pos_weight = nn.Parameter(data=torch.zeros(1), requires_grad=True)
            self.section_weight = nn.Parameter(data=torch.zeros(1), requires_grad=True)

        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)

    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional embedding to sequence of embeddings.

        Args:
            x:  Batch of embeddings as (batch, sequence, channels).
            shape:  Actual spatial shape of the input as (height, width).
        
        Returns:
            x:  Batch of embeddings as (batch, sequence, channels).

        Note: Interpolation of positional embeddings not implemented.
        """
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError(
                'Maximum requested position index exceeds the prepared position indices.'
            )
        # get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 1], :],
                      self.pos_embed[pos[:, :, 2], :]]
        # add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)

        # account for [CLS] token
        pos_embedding = torch.concatenate(
            [torch.zeros((B, 1, self.embed_dim), device=device), pos_embedding], dim=1,
        )
        # prepare cross-section embedding
        section_embedding = torch.zeros_like(pos_embedding, device=device)
        for i in range(B):
            # define the maximum number of cross-sections for a specimen
            # if the section embedding is not repeated, set the maximum to 
            # the embedding dimension
            if self.repeat_section_embedding:
                max_sections = int(torch.max(pos[i, :, 0])) 
            else:
                max_sections = self.embed_dim
            if max_sections > 0:
                for j in range(S):
                    section_embedding[i, j+1, pos[i, j, 0]::max_sections+1] = 1

        # flip the section embedding to start from the low frequency side
        section_embedding = torch.flip(section_embedding, (2,))

        # sum the positional and cross-section embedding
        if self.learnable_weight:
            combined_embedding = ((self.pos_weight*pos_embedding) 
                                  + (self.section_weight*section_embedding))
        else:
            combined_embedding = pos_embedding + section_embedding

        # check if the shape of the features and positional embeddings match
        if x.shape != combined_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+combined_embedding)
        
        return x


class PosEmbeddingFactory():

    def __init__(
        self, 
        embed_dim: int,
        expected_shape: Optional[tuple[int, int]] = None,
        dropout_prob: float = 0.0,
        learnable_weight: bool = False,
    ) -> None:
        """
        Returns positional embedder configured based on arguments.
        
        Args:
            embed_dim:  Dimensionality of embeddings.
            expected_shape:  Expected spatial shape of embeddings before sequence.
            dropout_prob:  Probablity of dropout.
            learnable_weight:  Indicates whether the fixed positional embedding 
                and section embedding are first multiplied with a learnable weight
                before being added to the feature vectors.
        """
        # select positional embedder implementation
        if expected_shape is not None:
            self.pos_embedder = LearnablePosEmbedding(
                embed_dim=embed_dim,
                expected_shape=expected_shape,
                dropout_prob=dropout_prob,
            )
        else:
            self.pos_embedder = PosEmbedding(
                embed_dim=embed_dim,
                dropout_prob=dropout_prob,
                learnable_weight=learnable_weight,
            )

    def get_positional_embedder(self):
        return self.pos_embedder


class PytorchMultiheadAttention(nn.MultiheadAttention):
    """ 
    Wrapper around nn.MultiheadAttention for making the key, value, and query 
    all equal to the input x of the forward method and set average_attn_weights=False.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, 
                 device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, 
                         add_zero_attn, kdim, vdim, batch_first, device, dtype)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, x, x, need_weights=True, average_attn_weights=False)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Initialize multi-head self-attention layer.

        Args:
            embed_dim:  Dimensionality of embedding used throughout the model.
            num_heads:  Number of heads in each multi-head attention sub-layer.
            qkv_bias:  Indicates whether biases are added to the queries, keys, 
                and values.
            attn_drop:  Probablity of dropout for multi-head attention.
            proj_drop:  Probablity of dropout for fully-connected layer.
        """
        super().__init__()
        # initialize instance attributes
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads)**-0.5
        
        # initialize layers
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        # get the queries, keys, and values
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # get attention weighted value embeddings and project them
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
        

class MultiHeadAttentionFactory():

    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        attn_dropout_prob: float,
        pytorch_imp: bool,
    ) -> None:
        """
        Returns multi-head attention implementation of the specified type with the
        correct configuration of input arguments.

        Notes:
        (1) The naive implementation has an option to enable or disable the bias 
            of the first linear layer, which by default was disabled. The Pytorch
            implementation only has an option to enable or disable the bias of
            both linear layers, which by default is enabled.
        
        Args:
            embed_dim:  Dimensionality of embeddings.
            n_heads:  Number of heads in multi-head attention.
            attn_dropout_prob:  Probability of dropping values from attention map.
            pytorch_imp:  Indicates if the Pytorch multi-head self-attention is used.
        """
        # select the correct implementation based on the requirements
        if pytorch_imp:                    
            self.multi_head_attention = PytorchMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                add_bias_kv=False,
                dropout=attn_dropout_prob,
                batch_first=True,
            )
        else:
            self.multi_head_attention = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                qkv_bias=True,
                attn_drop=attn_dropout_prob,
                proj_drop=0,
            )
    
    def get_multi_head_attention(self):
        return self.multi_head_attention


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, 
                           a: float, b: float) -> torch.Tensor:
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, 
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)