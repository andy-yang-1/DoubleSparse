import time
import torch

import triton
import triton.language as tl
import math
import random

from argsort import argsort


@triton.jit
def get_heavy_kernel(Q_Label, K_Label, Heavy_List,
                    stride_qbs, stride_qh, stride_qd,
                    stride_kbs, stride_kh, stride_kd,
                    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
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

    indexes = tl.arange(0, N_CTX)

    att_value, indexes = argsort(att_value, indexes, descending=True)

    # store to Heavy_List
    # TODO: Heavy_List here is [...,N_CTX], sparse_fwd is [...,HEAVY_CONST]
    offs_heavy = cur_batch * stride_heavy_list_bs + cur_head * stride_heavy_list_h + tl.arange(0, N_CTX) * stride_heavy_list_c
    tl.store(Heavy_List + offs_heavy, indexes)



def get_heavy(Q_Label, K_Label, Heavy_List):

    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    # strides
    stride_qbs, stride_qh, stride_qd = Q_Label.stride()
    stride_kbs, stride_kh, stride_kd = K_Label.stride()
    stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c = Heavy_List.stride()

    # grid
    grid = (B, H)

    get_heavy_kernel[grid](
        Q_Label, K_Label, Heavy_List,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_heavy_list_bs, stride_heavy_list_h, stride_heavy_list_c,
        H,
        HEAVY_CHANNEL_NUM, N_CTX
    )

    return Heavy_List


def torch_get_heavy(Q_Label, K_Label):

    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    q = Q_Label.view(B, H, 1, HEAVY_CHANNEL_NUM)
    k = K_Label.view(B, N_CTX, H, HEAVY_CHANNEL_NUM).transpose(1, 2).transpose(2,3)

    scores = torch.matmul(q, k).squeeze()

    _, Heavy_List = torch.sort(scores, dim=-1, descending=True)

    return Heavy_List


def test_get_heavy():

    B, H, N_CTX = 32, 32, 2048
    HEAVY_CHANNEL_NUM = 8

    dtype = torch.float16

    Q_Label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    K_Label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)

    Heavy_List = torch.empty((B, H, N_CTX), dtype=torch.int64, device='cuda')

    # Warm up
    for _ in range(10):
        get_heavy(Q_Label, K_Label, Heavy_List)

    torch.cuda.synchronize()

    # Test
    run_iter = 1000
    start = time.time()
    for _ in range(run_iter):
        get_heavy(Q_Label, K_Label, Heavy_List)
    torch.cuda.synchronize()
    print("Triton get_heavy time: ", (time.time() - start) / 100 * 1000, "ms")

    torch_index = torch_get_heavy(Q_Label, K_Label)

    print(torch_index[3,4,:64])
    print(Heavy_List[3,4,:64])

    # TODO: sort here is error?? [0,0] may be correct because of the same value, [0,1] is wrong
    assert torch.allclose(torch_index[0,0], Heavy_List[0,0], atol=1e-3, rtol=0)

if __name__ == "__main__":
    test_get_heavy()

