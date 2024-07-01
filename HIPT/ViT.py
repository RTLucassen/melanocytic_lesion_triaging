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

Which was copied from the original HIPT implementation by Chen et al.:
https://github.com/mahmoodlab/HIPT

Which was copied from the original DINO implementation by Caron et al.:
https://github.com/facebookresearch/dino

Which was mostly copied from the timm library by Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from HIPT.ViT_utils import (
    MultiHeadAttentionFactory, 
    PosEmbeddingFactory,
    trunc_normal_,
)


class DropPath(nn.Module):

    def __init__(self, drop_prob: Optional[float]) -> None:
        """
        Drop paths (Stochastic Depth) per sample 
        (when applied in main path of residual blocks).

        Args:
            drop_prob:  Probability of dropping path. 
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        else:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


class MLP(nn.Module):
   
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: torch.nn,
        dropout_prob: float,
    ) -> None:
        """
        Initialize multi-layer perceptron (MLP) for Transformer block.

        Args:
            in_features:  Dimensionality of input feature vector.
            hidden_features:  Dimensionality of hidden feature vector.
            out_features:  Dimensionality of output feature vector.
            act_layer:  Activation layer type.
            dropout_prob:  Probablity of dropout for fully-connected layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderLayer(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: Union[int, float],
        dropout_prob: float,
        attn_dropout_prob: float,
        drop_path_prob: float,
        act_layer: torch.nn,
        norm_layer: torch.nn,
        pytorch_attn_imp: bool,
    ) -> None:
        """
        Initialize Transformer encoder block.

        Args:
            embed_dim:  Dimensionality of embedding used throughout the model.
            n_heads:  Number of heads in each multi-head attention sub-layer.
            mlp_ratio:  Ratio between hidden and input/output feature vector 
                length of MLP.
            dropout_prob:  Probablity of dropout for positional embedding and 
                fully-connected layers.
            attn_dropout_prob:  Probablity of dropout for multi-head attention.
            drop_path_prob:  Probability of dropping path. 
            act_layer:  Activation layer type.
            norm_layer:  Normalization layer type.
            pos_embed_config:  Dictionary with positional embedding configuration. 
            pytorch_attn_imp:  Indicates whether the Pytorch multi-head 
                self-attention implementation is used.
        """
        super().__init__()
        # initialize first sublayer (multi-head attention)
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttentionFactory(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_dropout_prob=attn_dropout_prob,
            pytorch_imp=pytorch_attn_imp,
        ).get_multi_head_attention()
        
        # initialize stochastic depth
        if drop_path_prob > 0.0:
            self.drop_path = DropPath(drop_path_prob)  
        else:
            self.drop_path = nn.Identity()
        
        # initialize second sublayer (mlp)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            act_layer=act_layer,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor, return_self_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_self_attention:
            return x, attn
        else:
            return x


class ViT(nn.Module):

    def __init__(
        self,
        input_shape: Optional[Union[int, tuple[int, int]]],
        patch_shape: Optional[Union[int, tuple[int, int]]],
        input_dim: int,
        embed_dim: int,
        patient_dim: Optional[int],
        n_classes: Optional[int],
        depth: int,
        n_heads: int,
        mlp_ratio: Union[int, float],
        pos_embed: bool,
        learnable_weight: bool = False,
        patient_token: bool = True,
        dropout_prob: float = 0.0,
        attn_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        pytorch_attn_imp: bool = True,
    ) -> None:
        """
        Implementation of Vision Transformer (ViT).

        Args:
            input_shape: Expected spatial size of input (if None, no strict input 
                size is assumed and learnable positional embeddings can therefore 
                not be used).
            patch_shape:  Spatial size of patches (if None, an aggregation layer 
                is used instead of the patch embedding layer, i.e. Conv2d).
            input_dim:  Dimensionality of input embedding (for aggregation layers)
                or the number of input channels (for patch embedding layer).
            embed_dim:  Dimensionality of embedding used throughout the model.
            patient_dim:  Dimensionality of input feature vectors with patient 
                information (if None, no patient input feature vectors are used).
            n_classes:  Number of classes for prediction (if None or <1, 
                no classification layer is used).
            depth:  Number of transformer encoder layers.
            n_heads:  Number of heads in each multi-head attention sub-layer.
            mlp_ratio:  Ratio between hidden and input/output feature vector 
                length of MLP.
            pos_embed:  Indicates whether positional embeddings are used.
            learnable_weight:  Indicates whether the fixed positional encoding 
                and section encoding are first multiplied with a learnable weight
                before being added to the feature vectors.
            patient_token:  Indicates whether the patient information vector
                is incorporated as an additional token (after upsampling to embed_dim)
                or concatenated to the final [CLS] token before the final 
                classification layer.
            dropout_prob:  Probablity of dropout for positional embedding and 
                fully-connected layers.
            attn_dropout_prob:  Probablity of dropout for multi-head attention.
            drop_path_rate:  Maximum probability for dropping paths. 
            act_layer:  Activation layer type.
            norm_layer:  Normalization layer type.
            pytorch_attn_imp:  Indicates whether the Pytorch multi-head 
                self-attention is used.
        """
        super().__init__()

        # initialize instance attributes
        self.input_shape = input_shape
        if isinstance(self.input_shape, int):
            self.input_shape = (self.input_shape, self.input_shape)  
        self.patch_shape = patch_shape
        if isinstance(self.patch_shape, int):
            self.patch_shape = (self.patch_shape, self.patch_shape)            
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patient_dim = patient_dim
        if self.patient_dim is not None:
            if self.patient_dim < 1:
                self.patient_dim = None
        self.n_classes = n_classes
        self.depth = depth
        self.n_heads = n_heads
        self.pos_embed = pos_embed
        self.learnable_weight = learnable_weight
        self.patient_token = patient_token
        self.mlp_ratio = mlp_ratio
        self.dropout_prob = dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.pytorch_attn_imp = pytorch_attn_imp

        # determine number of added tokens (first is [CLS] token)
        if self.patient_dim is None or not self.patient_token:
            self.added_tokens = (1, 0) 
        else:
            self.added_tokens = (1, 1)

        # initialize patch embedding layer or aggregation layers
        if self.patch_shape is not None:
            self.patch_embed_layer = nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=self.embed_dim,
                kernel_size=self.patch_shape,
                stride=self.patch_shape,
            )
        else:
            self.aggregation_layers = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim), 
                self.act_layer(), 
                nn.Dropout(self.dropout_prob)
            )
        # initialize cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # determine the expected shape 
        if self.input_shape is not None:
            if self.patch_shape is not None:
                self.expected_shape = (
                    (self.input_shape[0]//self.patch_shape[0]),
                    (self.input_shape[1]//self.patch_shape[1]),
                )
            else:
                self.expected_shape = self.input_shape
        else:
            self.expected_shape = None

        # get positional embedder (optional)
        if self.pos_embed:
            self.pos_embedder = PosEmbeddingFactory(
                embed_dim=self.embed_dim,
                expected_shape=self.expected_shape,
                dropout_prob=self.dropout_prob,
                learnable_weight=self.learnable_weight,
            ).get_positional_embedder()

        # initialize linear layer, activation layer, and dropout layer for 
        # patient input feature vectors
        if self.patient_dim is not None or not self.patient_token:
            self.patient_layers = nn.Sequential(
                nn.Linear(self.patient_dim, self.embed_dim), 
                self.act_layer(), 
                nn.Dropout(self.dropout_prob)
            )
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=self.embed_dim,
                    n_heads=self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout_prob=self.dropout_prob,
                    attn_dropout_prob=self.attn_dropout_prob,
                    drop_path_prob=dpr[i],
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    pytorch_attn_imp=self.pytorch_attn_imp,
                )
                for i in range(depth)
            ]
        )
        self.norm = self.norm_layer(self.embed_dim)

        # initialize classifier head
        if self.n_classes is None:
            self.classifier = nn.Identity()
        elif self.n_classes < 1:
            self.classifier = nn.Identity()
        elif self.patient_dim is not None and not self.patient_token:
            self.classifier = nn.Linear(self.embed_dim + self.patient_dim,
                                        self.n_classes)
        else:
            self.classifier = nn.Linear(self.embed_dim, self.n_classes)

        # initialize parameter values
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_parameters)

    def _init_parameters(self, layer: torch.nn) -> None:
        """
        Initialize parameter values for layers in ViT.

        Args:
            layer: neural network layer.
        """
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.weight, 1.0)
            nn.init.constant_(layer.bias, 0)

    def forward(
        self, 
        x: torch.Tensor,
        x_patient: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        return_last_self_attention: bool = False,
    ) -> torch.Tensor:
        """ 
        Implementation of forward pass.
        
        Args:
            x:  Batch of input images or embeddings as (batch, channels, height, width).
            x_patient:  Batch of patient input feature vectors as (batch, features).
            pos:  Positions of elements in the sequence as (batch, sequence, position).
            return_last_self_attention:  Indicates whether the last self attention 
                values are returned.
        """
        # convert image to embeddings
        if self.patch_shape is not None:
            # [B, C_in, H_img, W_img] -> [B, C_out, H, W]
            # where H = (H_img//H_patch) and W = (W_img//W_patch)
            x = self.patch_embed_layer(x)
        
        if len(x.shape) == 3:
            input_shape = None
        elif len(x.shape) == 4:
            input_shape = x.shape[-2:]
            # convert to a sequence by combining the spatial dimensions if necessary
            # [B, C, H, W] -> [B, H*W, C] = [B, S, D]
            x = x.flatten(2, 3).transpose(1, 2)
        else:
            raise ValueError(f'Input has invalid number of dimensions: {len(x.shape)}')
        
        # apply aggregation layers to change the dimensionality
        if self.patch_shape is None:
            # [B, S, D_in] -> [B, S, D_out]
            x = self.aggregation_layers(x)  
        
        # create and add the [CLS] token to the sequence of embeddings 
        # (one for each item in the batch)
        # [1, 1, D] -> [B, 1, D]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, S, D] -> [B, 1+S, D]
        x = torch.cat((cls_tokens, x), dim=1)

        # (optionally) add positional embeddings
        if self.pos_embed:
            if self.expected_shape is not None:
                # [B, 1+S, D] -> [B, 1+S, D]
                x = self.pos_embedder(x, input_shape)
            elif pos is not None:
                # [B, 1+S, D] -> [B, 1+S, D]
                x = self.pos_embedder(x, pos)      
            else:
                raise ValueError(
                    'Invalid model configuration for adding positional embeddings. '
                    'Either specify `input_shape` at initialization or provide '
                    'the positions as argument to `pos` in the forward method.'
                )  

        # (optionally) add the patient token to the sequence of embeddings
        if x_patient is None:
            if self.patient_dim is not None:
                raise ValueError('A patient feature vector was expected but not '
                                'given as input argument.')
        elif self.patient_dim is not None and self.patient_token:
            # [B, D_in] -> [B, 1, D_out]
            x_patient = self.patient_layers(x_patient)[:, None, :]
            # [B, 1+S, D] -> [B, 1+S(+1), D]
            x = torch.cat((x, x_patient), dim=1)

        # feed embeddings to stacked transformer encoder layers
        for i, layer in enumerate(self.encoder_layers):
            # [B, 1+S(+1), D] -> [B, 1+S(+1), D]
            if return_last_self_attention and i == len(self.encoder_layers)-1:
                x, attn = layer(x, return_self_attention=True)
            else:
                x = layer(x)
        # [B, 1+S(+1), D] -> [B, 1+S(+1), D]
        x = self.norm(x)
        # get the [CLS] token
        # [B, 1+S(+1), D] -> [B, D]
        x = x[:, 0, :]

        # (optionally) add the patient feature vector to the 
        # classification feature vector
        if self.patient_dim is not None and not self.patient_token:
            # [B, D] -> [B, D(+D_patient)]
            x = torch.cat([x, x_patient], axis=1)

        # (optionally) make the classification prediction 
        # [B, D(+D_patient)] -> [B, n_classes]
        x = self.classifier(x)

        # return the final embedding or classification prediction 
        # and optionally the last self-attention values
        if return_last_self_attention:
            return x, attn
        else:
            return x

    def __repr__(self) -> str:
        """ 
        Print the layers and the total number of parameters.
        """
        n_params = 0
        n_params_train = 0
        for param in self.parameters():
            n = param.numel()
            n_params += n
            if param.requires_grad:
                n_params_train += n
        message = (
            f"{super().__repr__()}\n\n"
            f"Total number of parameters: {n_params}\n"
            f"Total number of trainable parameters: {n_params_train}"
        )
        return message


def convert_state_dict(
    path: Union[str, Path], 
    load_pos_embed: bool,
    pytorch_attn_imp: bool,
) -> dict:
    """
    Convert the state dict from the original HIPT model to our implementation.

    Args:
        path:  Path to state dict file.
        load_pos_embed:  Indicates whether the learned positional embeddings are used.
        pytorch_attn_imp:  Indicates whether the Pytorch multi-head 
            self-attention is used.
    """
    state_dict = torch.load(path, map_location='cpu')['teacher']
    original_layer_names = list(state_dict.keys())
    converted_state_dict = {}
    for name in original_layer_names:
        new_name = name.replace('module.', '').replace('backbone.', '')
        if 'cls_token' == new_name:
            converted_state_dict[new_name] = state_dict[name]
        elif 'pos_embed' == new_name:
            if load_pos_embed:
                new_name = 'pos_embedder.pos_embed'
                converted_state_dict[new_name] = state_dict[name]
        elif 'patch_embed' in new_name:
            new_name = new_name.replace('patch_embed.proj', 'patch_embed_layer')
            converted_state_dict[new_name] = state_dict[name]
        elif 'phi' in new_name:
            new_name = new_name.replace('phi', 'aggregation_layers')
            converted_state_dict[new_name] = state_dict[name]
        elif 'blocks' in new_name:
            new_name = new_name.replace('blocks', 'encoder_layers')
            if pytorch_attn_imp:
                if 'qkv' in new_name:
                    new_name = new_name.replace('qkv.', 'in_proj_')
                elif 'proj' in new_name:
                    new_name = new_name.replace('proj', 'out_proj')
                converted_state_dict[new_name] = state_dict[name]
            else:
                converted_state_dict[new_name] = state_dict[name]
        elif 'norm' in new_name:
            converted_state_dict[new_name] = state_dict[name]
        # skip all layers part of the DINO head
        elif 'head' in name:
            pass
        else:
            raise ValueError(f'Unrecognized layer name: {name}.')

    return converted_state_dict