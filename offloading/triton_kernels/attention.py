import torch
import sys

import triton
import triton.language as tl
import math

from channel import get_label_tensor
from sparse import fwd_sparse, torch_fwd_sparse
from heavy import get_heavy
from bgemv import bgemv
from bgemv_int8 import bgemv_int8


def att(Q, K, V, Out, q_label, k_label, label_scores, channel, heavy_const, heavy_channel_num, label_mask, attn_mask):

    get_label_tensor(Q, channel, q_label, heavy_channel_num)

    # get_label_tensor(K, channel, k_label, heavy_channel_num)

    # bgemv(q_label, k_label, label_scores)

    tmp_scores = torch.matmul(q_label.view(Q.shape[0], 1, Q.shape[1], heavy_channel_num).transpose(1,2), k_label.view(Q.shape[0], K.shape[0] // Q.shape[0], K.shape[1], heavy_channel_num).transpose(1,2).transpose(2, 3)).view(Q.shape[0], K.shape[1], K.shape[0] // Q.shape[0])

    # assert torch.allclose(tmp_scores, label_scores, atol=1e-4, rtol=0)

    _, label_index = torch.topk(tmp_scores, heavy_const, dim=-1)

    fwd_sparse(Q, K, V, Out, label_index, attn_mask)

    return Out

def att_int8(Q, K, V, Out, q_label, k_label, k_scales, label_scores, channel, heavy_const, heavy_channel_num, label_mask, attn_mask):

    get_label_tensor(Q, channel, q_label, heavy_channel_num)

    bgemv_int8(q_label, k_label, k_scales, label_scores)

    _, label_index = torch.topk(label_scores, heavy_const, dim=-1)

    fwd_sparse(Q, K, V, Out, label_index, attn_mask)

    return Out


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim, channel, heavy_const, label_mask, attn_mask):
    q = xq.view(bs, 1, num_head, head_dim)
    k = xk.view(bs, seqlen, num_head, head_dim)
    v = xv.view(bs, seqlen, num_head, head_dim)


    sorted_query_states = torch.gather(q, -1, channel.unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1)).transpose(1,2)
    sorted_key_states = torch.gather(k, -1, channel.unsqueeze(0).unsqueeze(0).expand(bs, seqlen, -1, -1)).transpose(1,2)

    label_scores = torch.matmul(sorted_query_states, sorted_key_states.transpose(2, 3))

    _, indices = torch.sort(label_scores, dim=-1, descending=True)

    discarded_indices = indices[:, :, :, heavy_const:]

    attn_weights = torch.matmul(q.transpose(1,2), k.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim)

    h2_mask = torch.zeros_like(attn_weights).bool()
    h2_mask.scatter_(dim=-1, index=discarded_indices, value=True)
    attn_weights.masked_fill_(h2_mask, float('-inf'))

    # bsz, num_head, 1, seqlen
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # tmp_output = torch.matmul(attn_weights, v.transpose(1,2)).transpose(1,2).contiguous().reshape(bs, 1, num_head * head_dim)
    attn_output = torch.matmul(attn_weights, v.transpose(1,2)).transpose(1,2).contiguous().reshape(bs, 1, num_head * head_dim)
    
    # assert torch.allclose(tmp_output.view(bs, num_head, head_dim), attn_output, atol=1e-2, rtol=0)

    return attn_output




def test_att(B, N_CTX, H, D, HEAVY_CHANNEL_NUM, HEAVY_CONST):
    import time

    # B, N_CTX, H, D = 1, 16384, 32, 128

    # HEAVY_CHANNEL_NUM = 8
    # HEAVY_CONST = 1024

    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CHANNEL_NUM: {HEAVY_CHANNEL_NUM}, HEAVY_CONST: {HEAVY_CONST}")

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)

    channel = torch.zeros(H, HEAVY_CHANNEL_NUM, dtype=torch.int64, device='cuda')
    for h in range(H):
        channel[h] = torch.randperm(D, device='cuda')[:HEAVY_CHANNEL_NUM]

    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    q_label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")
    k_label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")

    label_scores = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")

    heavy_list = torch.empty((B, H, N_CTX), dtype=torch.int64, device='cuda')

    get_label_tensor(k, channel, k_label, HEAVY_CHANNEL_NUM)

    label_mask = torch.zeros((B, N_CTX), dtype=dtype, device="cuda")
    attn_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")

    # k_label = k_label.view(B, N_CTX, H, HEAVY_CHANNEL_NUM).transpose(1, 2).transpose(2,3)

    # global att
    # att = torch.compile(att, fullgraph=True) # mode limited

    # Warm up
    for _ in range(10):
        att(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask)

    
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        att(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    torch_out = torch_att(q, k, v, B, N_CTX, H, D, channel, HEAVY_CONST, label_mask, attn_mask).squeeze().view(B, H, D)
    # att(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, mask)
    o = out


    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)

    return (t2 - t1) / run_iter


def test_att_int8(B, N_CTX, H, D, HEAVY_CHANNEL_NUM, HEAVY_CONST):
    import time

    # B, N_CTX, H, D = 1, 16384, 32, 128

    # HEAVY_CHANNEL_NUM = 8
    # HEAVY_CONST = 1024

    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CHANNEL_NUM: {HEAVY_CHANNEL_NUM}, HEAVY_CONST: {HEAVY_CONST}")

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)

    channel = torch.zeros(H, HEAVY_CHANNEL_NUM, dtype=torch.int64, device='cuda')
    for h in range(H):
        channel[h] = torch.randperm(D, device='cuda')[:HEAVY_CHANNEL_NUM]

    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    q_label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")
    k_label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")

    label_scores = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")

    heavy_list = torch.empty((B, H, N_CTX), dtype=torch.int64, device='cuda')

    get_label_tensor(k, channel, k_label, HEAVY_CHANNEL_NUM)

    label_mask = torch.zeros((B, N_CTX), dtype=dtype, device="cuda")
    attn_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")

    k_scales = (k_label.abs().max(-1)[0] / 127.0)
    k_label = (k_label / k_scales[:, :, None]).to(torch.int8)

    # warm up
    for _ in range(10):
        att_int8(q, k, v, out, q_label, k_label, k_scales, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask)
    
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        att_int8(q, k, v, out, q_label, k_label, k_scales, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask)
    torch.cuda.synchronize()

    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    return (t2 - t1) / run_iter


if __name__ == '__main__':

    # bszs = [1, 4, 8, 16, 32]
    # ctxs = [2048, 4096, 8192, 16384]

    bsz = int(sys.argv[1])
    ctx = int(sys.argv[2])

    sparsity_level = 16
    h = 32
    d = 128


    # times = []

    att = torch.compile(att, fullgraph=True) # mode limited
    print(f"bsz: {bsz}, ctx: {ctx}, time: {test_att(bsz, ctx, h, d, d // sparsity_level, ctx // sparsity_level)}")


    # for b in bszs:
    #     for n_ctx in ctxs:
    #         heavy_channel_num = d // sparsity_level
    #         heavy_const = n_ctx // sparsity_level
    #         # test_att(b, n_ctx, h, d, heavy_channel_num, heavy_const)
    #         times.append([b, n_ctx, test_att(b, n_ctx, h, d, heavy_channel_num, heavy_const)])

    # print(times)
