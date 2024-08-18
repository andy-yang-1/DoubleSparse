import torch
import torch.nn as nn
from torch.nn import functional as F


def test_flash(B, N_CTX, H, D):
    import time

    # B, N_CTX, H, D = 1, 16384, 32, 128

    dtype = torch.float16

    q = torch.empty((B, 1, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B, N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B, N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    att_out = torch.empty((H, B * N_CTX), dtype=dtype, device="cuda")
    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    # Warm up
    for _ in range(10):
        y = F.scaled_dot_product_attention(q, k, v)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        y = F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    # torch_out = torch_att(q, k, v, B, N_CTX, H, D).squeeze()
    o = out

    # print("max ", torch.max(torch.abs(torch_out - o)))
    # print("mean ", torch.mean(torch.abs(torch_out - o)))
    # assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)
    return (t2 - t1) / run_iter

if __name__ == "__main__":

    bszs = [1, 4, 8, 16, 32]
    ctxs = [2048, 4096, 8192, 16384]

    h = 32
    d = 128

    times = []

    for b in bszs:
        for n_ctx in ctxs:
            times.append([b, n_ctx, test_flash(b, n_ctx, h, d)])

    print(times)