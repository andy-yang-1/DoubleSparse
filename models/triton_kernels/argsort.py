import numpy as np
import torch
import triton


# modified from triton.language.standard


@triton.jit
def custom_compare_and_swap(x, indexes, desc_mask, n_dims: triton.language.constexpr, idx: triton.language.constexpr):
    x_int = triton.language.standard._cast_to_int(x)
    l_int = triton.language.standard._take_slice(x_int, n_dims, idx, 0)
    r_int = triton.language.standard._take_slice(x_int, n_dims, idx, 1)
    l = l_int.to(x.dtype, bitcast=True)
    r = r_int.to(x.dtype, bitcast=True)

    l_idx = triton.language.standard._take_slice(indexes, n_dims, idx, 0)
    r_idx = triton.language.standard._take_slice(indexes, n_dims, idx, 1)

    desc_mask = desc_mask.to(x_int.dtype)
    zero = triton.language.zeros_like(x_int)
    swap = (l > r) ^ desc_mask

    y = x_int ^ triton.language.where(swap, l_int ^ r_int, zero)
    indexes_y = indexes ^ triton.language.where(swap, l_idx ^ r_idx, triton.language.zeros_like(indexes))

    y = y.to(x.dtype, bitcast=True)
    return y, indexes_y


@triton.jit
def custom_bitonic_merge(x, indexes, n_dims: triton.language.constexpr, active_dims: triton.language.constexpr, order_type: triton.language.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    triton.language.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = triton.language.standard._indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in triton.language.static_range(active_dims):
        x, indexes = custom_compare_and_swap(x, indexes, desc_mask, n_dims, active_dims - 1 - i)

    return x, indexes


@triton.jit
def argsort(x, indexes, dim=None, descending: triton.language.constexpr = 0):
    triton.language.static_assert(triton.language.standard._is_power_of_two(x.shape[triton.language.standard._get_sort_dim(dim, x.shape)]))
    triton.language.static_assert(triton.language.standard._is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = triton.language.reshape(x, [2] * triton.language.standard._log2(x.numel))
    y_indexes = triton.language.reshape(indexes, [2] * triton.language.standard._log2(x.numel))
    for i in triton.language.static_range(1, triton.language.standard._log2(x.shape[triton.language.standard._get_sort_dim(dim, x.shape)]) + 1):
        y, y_indexes = custom_bitonic_merge(y, y_indexes, triton.language.standard._log2(x.numel), i, (descending if
                                                  (i == triton.language.standard._log2(x.shape[triton.language.standard._get_sort_dim(dim, x.shape)])) else 2))

    x = triton.language.reshape(y, x.shape)
    indexes = triton.language.reshape(y_indexes, indexes.shape)
    return x, indexes


@triton.jit
def sort_kernel(X, Z, I, N: triton.language.constexpr, M: triton.language.constexpr, descending: triton.language.constexpr):
    offx = triton.language.arange(0, M)
    offy = triton.language.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = triton.language.load(X + off2d)
    indexes = triton.language.arange(0,M)[None,:]
    indexes = triton.language.broadcast_to(indexes, [N, M])
    x, indexes = argsort(x, indexes, descending=descending)
    # x = triton.language.sort(x, descending=descending)
    triton.language.store(Z + off2d, x)
    triton.language.store(I + off2d, indexes)


def test_argsort():

    M = 256
    N = 8
    x = np.random.rand(N, M).astype(np.float32)
    x = torch.from_numpy(x).to("cuda")
    y, i0 = torch.sort(x, descending=True)
    z = torch.empty_like(x)
    i = torch.empty_like(i0)
    pgm = sort_kernel[(1, )](x, z, i, N, M, True, num_warps=8)
    assert (y == z).all(), (y, z)
    assert (i0 == i).all(), (i0, i)


if __name__ == "__main__":
    test_argsort()

