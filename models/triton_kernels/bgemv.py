import time
import torch

import triton
import triton.language as tl
import math
import random


@triton.jit
def bgemv_kernel(Q_Label, K_Label, Out,
                    stride_qbs, stride_qh, stride_qd,
                    stride_kbs, stride_kh, stride_kd,
                    stride_out_bs, stride_out_h, stride_out_c,
                    # B: tl.constexpr, H: tl.constexpr,
                    BLOCK_HMODEL: tl.constexpr,
                    HEAVY_CHANNEL_NUM: tl.constexpr,
                    N_CTX: tl.constexpr):
    
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # [0:HEAVY_CHANNEL_NUM]
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + tl.arange(0, HEAVY_CHANNEL_NUM) * stride_qd

    # [0:N_CTX,0:HEAVY_CHANNEL_NUM]
    offs_k = cur_batch * N_CTX * stride_kbs + tl.arange(0, N_CTX)[:, None] * BLOCK_HMODEL * stride_kh + cur_head * stride_kh + tl.arange(0, HEAVY_CHANNEL_NUM)[None, :] * stride_kd


    # load q k
    q = tl.load(Q_Label + offs_q)
    k = tl.load(K_Label + offs_k)

    # compute att
    att_value = tl.sum(q[None, :] * k, 1)


    # store to Out: [B, H, N_CTX]
    offs_out = cur_batch * stride_out_bs + cur_head * stride_out_h + tl.arange(0, N_CTX) * stride_out_c
    tl.store(Out + offs_out, att_value)


def bgemv(Q_Label, K_Label, Out):

    # TODO: now support N_CTX == 2**n only
    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    stride_qbs, stride_qh, stride_qd = Q_Label.stride()
    stride_kbs, stride_kh, stride_kd = K_Label.stride()
    stride_out_bs, stride_out_h, stride_out_c = Out.stride()

    grid = (B, H)

    bgemv_kernel[grid](
        Q_Label, K_Label, Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_out_bs, stride_out_h, stride_out_c,
        H,
        HEAVY_CHANNEL_NUM, N_CTX
    )

    return Out


def torch_bgemv(Q_Label, K_Label):

    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    q = Q_Label.view(B, H, 1, HEAVY_CHANNEL_NUM)
    k = K_Label.view(B, N_CTX, H, HEAVY_CHANNEL_NUM).transpose(1, 2).transpose(2,3)

    scores = torch.matmul(q, k).squeeze(-2)

    return scores


def test_bgemv():

    B, H, N_CTX = 32, 32, 2048
    HEAVY_CHANNEL_NUM = 8

    dtype = torch.float16

    Q_Label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    K_Label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    Out = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")


    # Warm up
    for _ in range(10):
        bgemv(Q_Label, K_Label, Out)

    torch.cuda.synchronize()

    # Test
    run_iter = 1000
    start = time.time()
    for _ in range(run_iter):
        bgemv(Q_Label, K_Label, Out)
    torch.cuda.synchronize()
    print("Triton bgemv time: ", (time.time() - start) / run_iter)

    torch_out = torch_bgemv(Q_Label, K_Label)

    # TODO: bgemv precision problem?
    print("max ", torch.max(torch.abs(torch_out - Out)))
    print("mean ", torch.mean(torch.abs(torch_out - Out)))
    assert torch.allclose(torch_out, Out, atol=1e-3, rtol=0)

if __name__ == "__main__":
    test_bgemv()
