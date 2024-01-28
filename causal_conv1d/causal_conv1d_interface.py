# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn.functional as F


import causal_conv1d_cuda


class CausalConv1dFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None, seq_idx=None, activation=None, cache_conv_state=None):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        cache_conv_state = cache_conv_state.contiguous() if cache_conv_state is not None else None
        ctx.save_for_backward(x, weight, bias, seq_idx, cache_conv_state)
        ctx.activation = activation in ["silu", "swish"]
        out, new_cache_conv_state = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, seq_idx, ctx.activation, cache_conv_state=cache_conv_state)
        return out, new_cache_conv_state

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, seq_idx, cache_conv_state = ctx.saved_tensors
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, weight, bias, dout, seq_idx, None, ctx.activation, cache_conv_state=cache_conv_state,
        )
        return dx, dweight, dbias if bias is not None else None, None, None


def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None, cache_conv_state=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    activation: either None or "silu" or "swish"
    cache_conv_state: (batch, dim, width2)
        width2 = width - 1

    out: (batch, dim, seqlen)
    """
    return CausalConv1dFn.apply(x, weight, bias, seq_idx, activation, cache_conv_state=cache_conv_state)


def causal_conv1d_ref(x, weight, bias=None, activation=None, cache_conv_state=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    cache_conv_state: (batch, dim, width2)
        width2 = width - 1

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    dtype_in = x.dtype
    seqlen = x.shape[-1]
    dim, width = weight.shape
    width2 = cache_conv_state.shape[-1]
    
    assert width2 == width - 1, "width2 != width - 1, width2 = {}, width = {}".format(width2, width)


    # print("In causal_conv1d_ref")
    # print("seqlen:", seqlen)
    # print("width:", width)
    # print("width2:", width2)


    if cache_conv_state is not None:
        x = torch.cat([cache_conv_state, x], dim=-1) # (batch, dim, width2 + seqlen)
        x = x.to(weight.dtype)
        new_cache_conv_state = x[:, :, -width2:]
    else:
        assert cache_conv_state is None, "cache_conv_state must be None if not provided"
        x = x.to(weight.dtype)
        new_cache_conv_state = None
    
    out = F.conv1d(x, weight.unsqueeze(1), bias, groups=dim)
    print("seqlen is ", seqlen)
    print("out.shape:", out.shape)
    assert out.shape[-1] == seqlen, "out.shape[-1] != seqlen, out.shape[-1] = {}, seqlen = {}".format(out.shape[-1], seqlen)


    out = out[..., :seqlen]

    return (out if activation is None else F.silu(out)).to(dtype=dtype_in), new_cache_conv_state


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]
    return causal_conv1d_cuda.causal_conv1d_update(x, conv_state, weight, bias, activation)


def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    batch, dim = x.shape
    width = weight.shape[1]
    assert conv_state.shape == (batch, dim, width)
    assert weight.shape == (dim, width)
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1)) # Update state (B D W)
    conv_state[:, :, -1] = x
    out = torch.sum(conv_state * weight, dim=-1) # (B D)
    if bias is not None:
        out += bias
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)
