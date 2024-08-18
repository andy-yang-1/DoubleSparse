import torch
from torch import abs, softmax, sqrt, tensor, topk

import triton
import triton.language as tl
import math

from channel import get_label_tensor
from sparse import fwd_sparse, torch_fwd_sparse
from heavy import get_heavy
from bgemv import bgemv
from bgemv_int8 import bgemv_int8


def sparq_att(Q, K, V, Out, q_label, k_label, r, k, mask):

    channel = topk(abs(Q), r, -1).indices[0]
    # print(channel.shape)

    get_label_tensor(Q, channel, q_label, r)
    get_label_tensor(K, channel, k_label, r)

    # bgemv(q_label, k_label, label_scores)

    tmp_scores = torch.matmul(q_label.view(Q.shape[0], 1, Q.shape[1], r).transpose(1,2), k_label.view(Q.shape[0], K.shape[0] // Q.shape[0], K.shape[1], r).transpose(1,2).transpose(2, 3)).view(Q.shape[0], K.shape[1], K.shape[0] // Q.shape[0])

    # assert torch.allclose(tmp_scores, label_scores, atol=1e-4, rtol=0)

    _, label_index = torch.topk(tmp_scores, k, dim=-1)

    fwd_sparse(Q, K, V, Out, label_index, mask)

    return Out



def test_sparq_att(B, N_CTX, H, D, r, k):
    import time

    # B, N_CTX, H, D = 1, 16384, 32, 128

    # HEAVY_CHANNEL_NUM = 8
    # HEAVY_CONST = 1024

    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, r: {r}, k: {k}")

    dtype = torch.float16

    Q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    K = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    V = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)


    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    q_label = torch.empty((B, H, r), dtype=dtype, device="cuda")
    k_label = torch.empty((B * N_CTX, H, r), dtype=dtype, device="cuda")

    mask = torch.zeros((B, k), dtype=dtype, device="cuda")

    # k_label = k_label.view(B, N_CTX, H, HEAVY_CHANNEL_NUM).transpose(1, 2).transpose(2,3)

    # Warm up
    for _ in range(10):
        sparq_att(Q, K, V, out, q_label, k_label, r, k, mask)

    
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        sparq_att(Q, K, V, out, q_label, k_label, r, k, mask)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))



    # print("max ", torch.max(torch.abs(torch_out - o)))
    # print("mean ", torch.mean(torch.abs(torch_out - o)))
    # assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)

    return (t2 - t1) / run_iter



if __name__ == '__main__':

    bszs = [1, 4, 8, 16, 32]
    ctxs = [2048, 4096, 8192, 16384]


    sparsity_level = 16
    h = 32
    d = 128


    times = []

    for b in bszs:
        for n_ctx in ctxs:
            heavy_channel_num = d // sparsity_level
            heavy_const = n_ctx // sparsity_level
            # test_att(b, n_ctx, h, d, heavy_channel_num, heavy_const)
            times.append([b, n_ctx, test_sparq_att(b, n_ctx, h, d, heavy_channel_num, heavy_const)])

    print(times)
