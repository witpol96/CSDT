from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer, l2norm
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights,LayerNorm
import torch
import torch.nn as nn 
import torch.nn.functional as F
from datasets.bases import tokenize
from utils.simple_tokenizer import SimpleTokenizer
import copy
import random
from collections import OrderedDict
import math

# 定义 LoRA 模块
class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 16,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_loras: int = 2,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out

        if lora_r > 0:
            self.loras_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r, in_features))) for _ in range(num_loras)])
            self.loras_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r))) for _ in range(num_loras)])
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for lora_A in self.loras_A:
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            for lora_B in self.loras_B:
                nn.init.zeros_(lora_B)

    def forward(self, x: torch.Tensor, lora_index=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0:
            result = F.linear(x, T(self.weight), bias=self.bias)
            lora_result = (self.lora_dropout(x) @ self.loras_A[lora_index].transpose(0, 1) @ self.loras_B[lora_index].transpose(0, 1)) * self.scaling
            result = result + lora_result
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlockwithoutLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, inputs: torch.Tensor,lora_index=0):
        x = inputs[0]
        atten, atten_weight = self.attention(self.ln_1(x))
        x = x + atten
        x = x + self.mlp(self.ln_2(x))
        return [x, atten_weight]

class ResidualAttentionBlockMLPwithLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4, lora_r=lora_r, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model, lora_r=lora_r, num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # def attention(self, x: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
    
    def forward(self, inputs: torch.Tensor, lora_index=0):
        x = inputs[0]
        atten, atten_weight = self.attention(self.ln_1(x))
        x = x + atten
        x_mlp = self.mlp[0](self.ln_2(x),lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp,lora_index)
        x = x + x_mlp
        return [x, atten_weight]


class MMTransformer_withlora(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp'):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers<=layers
        print('LoRA Mode:{}'.format(lora_mode))
        
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlockwithoutLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
            [ResidualAttentionBlockMLPwithLoRA(width, heads, lora_r, num_loras, attn_mask) for _ in range(lora_layers)])
        

    def forward(self, x: torch.Tensor, lora_index=0):
        # lora_mapping = {"RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
        #                 "NIR+CP":5,"NIR+SK":6,"NIR+TEXT":7,"CP+SK":8,"CP+TEXT":9,"SK+TEXT":10,
        #                 "NIR+CP+SK":11,"NIR+CP+TEXT":12,"NIR+SK+TEXT":13,"CP+SK+TEXT":14,
        #                 "NIR+CP+SK+TEXT":15}
        for block in self.resblocks:
            x = block(x, lora_index)
        return x
        # else:
    #     transformer = copy.deepcopy(base_model.visual.transformer)
    # ln_post = copy.deepcopy(base_model.visual.ln_post)
    # proj = copy.deepcopy(base_model.visual.proj)
    # encoder = MultimodalVisionEncoder(transformer,ln_post,proj)
    # return encoder



# class TextTransformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

