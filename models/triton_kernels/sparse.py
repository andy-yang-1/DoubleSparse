import time
import torch

import triton
import triton.language as tl
import math
import random

# from .argsort import argsort


@triton.jit
def fwd_sparse_kernel(
    Q, K, V, sm_scale, Heavy_List, Mask,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
    stride_mbs, stride_mc,
    out_stride_bs, out_stride_h, out_stride_d,

    N_CTX,
    HEAVY_CONST: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HMODEL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # [0:128]
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # q's offset -> [0:128]
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    # heavy list's offset -> [0:HEAVY_CONST]
    offs_heavy = cur_batch * stride_heavy_list_bs + cur_head * stride_heavy_list_h + tl.arange(0, HEAVY_CONST) * stride_heavy_list_c
    heavy_list = tl.load(Heavy_List + offs_heavy)

    # kv's offset -> [0:HEAVY_CONST,0:128]
    # batch -> heavy list -> head -> dmodel
    off_kv = cur_batch * N_CTX * stride_kbs + heavy_list[:, None] * BLOCK_HMODEL * stride_kh + cur_head * stride_kh + offs_d[None, :] * stride_kd

    off_mask = cur_batch * stride_mbs + tl.arange(0, HEAVY_CONST) * stride_mc

    # load q k v
    q = tl.load(Q + off_q)
    k = tl.load(K + off_kv)
    v = tl.load(V + off_kv)

    # load mask
    mask = tl.load(Mask + off_mask)

    # compute att
    att_value = tl.sum(q[None, :] * k, 1)
    att_value *= sm_scale
    att_value += mask
    attn_weight = tl.softmax(att_value)
    # attn_weight = tl.softmax(att_value.to(tl.float32)).to(tl.float16)
    att_out = tl.sum(attn_weight[:, None] * v, 0)

    # store to out
    off_out = cur_batch * out_stride_bs + cur_head * out_stride_h + offs_d * out_stride_d
    tl.store(Out + off_out, att_out)


def fwd_sparse(Q, K, V, Out, Heavy_List, Mask):

    Lq, Lk = Q.shape[-1], K.shape[-1]

    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    B, H, D = Q.shape
    HEAVY_CONST = Heavy_List.shape[-1]
    N_CTX = K.shape[0] // B

    # strides
    stride_qbs, stride_qh, stride_qd = Q.stride()
    stride_kbs, stride_kh, stride_kd = K.stride()
    stride_vbs, stride_vh, stride_vd = V.stride()
    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c = Heavy_List.stride()
    stride_mbs, stride_mc = Mask.stride()
    out_stride_bs, out_stride_h, out_stride_d = Out.stride()

    # grid
    grid = (B, H)

    fwd_sparse_kernel[grid](
        Q, K, V, sm_scale, Heavy_List, Mask,
        Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
        stride_mbs, stride_mc,
        out_stride_bs, out_stride_h, out_stride_d,
        N_CTX, HEAVY_CONST, D, H
    )

    return Out


@triton.jit
def fwd_sparse_no_mask_kernel(
    Q, K, V, sm_scale, Heavy_List,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
    out_stride_bs, out_stride_h, out_stride_d,

    N_CTX,
    HEAVY_CONST: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HMODEL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # [0:128]
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # q's offset -> [0:128]
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    # heavy list's offset -> [0:HEAVY_CONST]
    offs_heavy = cur_batch * stride_heavy_list_bs + cur_head * stride_heavy_list_h + tl.arange(0, HEAVY_CONST) * stride_heavy_list_c
    heavy_list = tl.load(Heavy_List + offs_heavy)

    # kv's offset -> [0:HEAVY_CONST,0:128]
    # batch -> heavy list -> head -> dmodel
    off_kv = cur_batch * N_CTX * stride_kbs + heavy_list[:, None] * BLOCK_HMODEL * stride_kh + cur_head * stride_kh + offs_d[None, :] * stride_kd

    # load q k v
    q = tl.load(Q + off_q)
    k = tl.load(K + off_kv)
    v = tl.load(V + off_kv)

    # compute att
    att_value = tl.sum(q[None, :] * k, 1)
    att_value *= sm_scale
    attn_weight = tl.softmax(att_value)
    att_out = tl.sum(attn_weight[:, None] * v, 0)

    # store to out
    off_out = cur_batch * out_stride_bs + cur_head * out_stride_h + offs_d * out_stride_d
    tl.store(Out + off_out, att_out)


def fwd_sparse_no_mask(Q, K, V, Out, Heavy_List):

    Lq, Lk = Q.shape[-1], K.shape[-1]

    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    B, H, D = Q.shape
    HEAVY_CONST = Heavy_List.shape[-1]
    N_CTX = K.shape[0] // B

    # strides
    stride_qbs, stride_qh, stride_qd = Q.stride()
    stride_kbs, stride_kh, stride_kd = K.stride()
    stride_vbs, stride_vh, stride_vd = V.stride()
    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c = Heavy_List.stride()
    out_stride_bs, out_stride_h, out_stride_d = Out.stride()

    # grid
    grid = (B, H)

    fwd_sparse_no_mask_kernel[grid](
        Q, K, V, sm_scale, Heavy_List,
        Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
        out_stride_bs, out_stride_h, out_stride_d,
        N_CTX, HEAVY_CONST, D, H
    )

    return Out


def torch_fwd_sparse(Q, K, V, Heavy_List, Mask):

    B, H, D = Q.shape

    Out = torch.zeros(B, H, D, dtype=Q.dtype, device='cuda')

    K = K.view(B, -1, H, D)
    V = V.view(B, -1, H, D)

    for b in range(B):
        for h in range(H):
            q = Q[b, h]
            heavy_list = Heavy_List[b, h]
            k = K[b, heavy_list, h]
            v = V[b, heavy_list, h]

            att_value = torch.sum(q[None, :] * k, 1)
            att_value *= 1.0 / (D ** 0.5)
            att_value += Mask[b]
            attn_weight = torch.softmax(att_value.to(torch.float32), 0).to(torch.float16)
            att_out = torch.sum(attn_weight[:, None] * v, 0)

            Out[b, h] = att_out

    return Out


def test_fwd_sparse():

    B, N_CTX, H, D = 32, 2048, 32, 128
    HEAVY_CONST = 128

    dtype = torch.float16

    Q = torch.randn(B, H, D, dtype=dtype, device='cuda').normal_(mean=0.1, std=0.2)
    K = torch.randn(B * N_CTX, H, D, dtype=dtype, device='cuda').normal_(mean=0.1, std=0.2)
    V = torch.randn(B * N_CTX, H, D, dtype=dtype, device='cuda').normal_(mean=0.1, std=0.2)

    Heavy_List = torch.zeros(B, H, HEAVY_CONST, dtype=torch.int64, device='cuda')
    Mask = torch.zeros(B, HEAVY_CONST, dtype=dtype, device='cuda')

    for b in range(B):
        for h in range(H):
            Heavy_List[b,h] = torch.randperm(N_CTX, device='cuda')[:HEAVY_CONST]

    Out = torch.zeros(B, H, D, dtype=dtype, device='cuda')

    # Warm up
    fwd_sparse(Q, K, V, Out, Heavy_List, Mask)

    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        fwd_sparse(Q, K, V, Out, Heavy_List, Mask)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time cost {(t2 - t1) / run_iter}")


    torch_out = torch_fwd_sparse(Q, K, V, Heavy_List, Mask)

    print("max ", torch.max(torch.abs(torch_out - Out)))
    print("mean ", torch.mean(torch.abs(torch_out - Out)))
    assert torch.allclose(torch_out, Out, atol=1e-3, rtol=0)


if __name__ == '__main__':
    test_fwd_sparse()