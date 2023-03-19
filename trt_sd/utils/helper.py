from types import MethodType as FunctionRewrite
from typing import Callable, List, Optional

import torch
from torch import Module, Tensor
from torch._C import Graph, Value
from torch.autograd import Function
from torch.autograd.function import BackwardCFunction
from torch.onnx import symbolic_helper


class MyCrossAttnProcessor:

    def __init__(self, forward: Callable) -> None:
        self.forward = forward

    def __call__(
        self,
        attn: Module,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        output = self.forward(self, attn, hidden_states, encoder_hidden_states,
                              attention_mask)
        return output


class VAEEncoderWrapper(Module):

    def __init__(self, vae: Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, sample: Tensor) -> Tensor:
        latent = self.vae.encode(sample)[0].sample()
        return latent


class VAEDecoderWrapper(Module):

    def __init__(self, vae: Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: Tensor) -> Tensor:
        return self.vae.decode(latents)


class SplitGeLU(Function):

    @staticmethod
    def symbolic(g: Graph, x: Value) -> Value:
        op = g.op('TRT::SplitGeLU', x)
        return op

    @staticmethod
    def forward(ctx: BackwardCFunction, x: Tensor) -> Tensor:
        _shape = x.shape[-1] // 2
        return x[..., :_shape]


class CrossAttn_V1(Function):

    @staticmethod
    def symbolic(g: Graph, x: Value) -> Value:
        op = g.op('TRT::fMHA_V2', x)
        return op

    @staticmethod
    def forward(ctx: BackwardCFunction, x: Tensor) -> Tensor:
        return x[..., 0, :]


class CrossAttn_V2(Function):

    @staticmethod
    def symbolic(g: Graph, q: Value, kv: Value) -> Value:
        op = g.op('TRT::fMHCA', q, kv)
        return op

    @staticmethod
    def forward(ctx: BackwardCFunction, q: Tensor, kv: Tensor) -> Tensor:
        return q


def split_gelu_forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = self.proj(hidden_states)
    return SplitGeLU.apply(hidden_states)


def split_gelu(module: Module) -> None:
    module.forward = FunctionRewrite(split_gelu_forward, module)


def cross_attn_forward_v1(self,
                          attn: Module,
                          hidden_states: Tensor,
                          encoder_hidden_states: Optional[Tensor] = None,
                          attention_mask: Optional[Tensor] = None) -> Tensor:
    nHead = attn.heads
    q = attn.to_q.weight.t().clone()
    k = attn.to_k.weight.t().clone()
    v = attn.to_v.weight.t().clone()
    fDim = q.shape[0]
    fuse_qkv = torch.cat([
        q.reshape(fDim, nHead, -1),
        k.reshape(fDim, nHead, -1),
        v.reshape(fDim, nHead, -1)
    ], -1).reshape(fDim, -1)
    b, c, *_ = hidden_states.shape
    fMHA_V2_input = (hidden_states @ fuse_qkv).reshape(b, c, nHead, 3, -1)
    fMHA_V2_output = CrossAttn_V1.apply(fMHA_V2_input)
    fMHA_V2_output = fMHA_V2_output.reshape(b, c, -1)
    output = attn.to_out[0](fMHA_V2_output)
    return output


def cross_attn_forward_v2(self,
                          attn: Module,
                          hidden_states: Tensor,
                          encoder_hidden_states: Optional[Tensor] = None,
                          attention_mask: Optional[Tensor] = None) -> Tensor:
    nHead = attn.heads
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.cross_attention_norm:
        encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
    # q = attn.to_q.weight.T.clone()
    query = attn.to_q(hidden_states)
    qb, qc, *_ = query.shape
    query = query.reshape(qb, qc, nHead, -1)

    k = attn.to_k.weight.t().clone()
    v = attn.to_v.weight.t().clone()

    fDim = k.shape[0]
    fuse_kv = torch.cat(
        [k.reshape(fDim, nHead, -1),
         v.reshape(fDim, nHead, -1)], -1).reshape(fDim, -1)
    b, c, *_ = encoder_hidden_states.shape
    _c = hidden_states.shape[1]
    fMHCA_input = (encoder_hidden_states @ fuse_kv).reshape(b, c, nHead, 2, -1)
    fMHCA_output = CrossAttn_V2.apply(query, fMHCA_input)
    fMHCA_output = fMHCA_output.reshape(b, _c, -1)
    output = attn.to_out[0](fMHCA_output)
    return output


@symbolic_helper.parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g: Graph, input: Value, normalized_shape: List, weight: Value,
               bias: Value, eps: float, cudnn_enable: Value) -> Value:
    axis = [-i for i in range(len(normalized_shape), 0, -1)]
    op = g.op(
        'TRT::LayerNorm',
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )
    return op


@symbolic_helper.parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g: Graph, input: Value, num_groups: int, weight: Value,
               bias: Value, eps: float, cudnn_enable: bool) -> Value:
    op = g.op(
        'TRT::GroupNorm',
        input,
        weight,
        bias,
        epsilon_f=eps,
        bSwish_i=0,
    )
    return op


def triu_onnx(x: Tensor, diagonal: int = 0) -> Tensor:
    _l = x.shape[0]
    arange = torch.arange(_l, device=x.device)
    mask = arange.expand(_l, _l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return mask * x
