import torch
import torch.nn.functional as F

def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def pseudo_quantize(tensor, q_bit):
    max_quant = 2 ** q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    dequantized = quantized / scale + min_val

    return dequantized


def quantize(tensor, q_bit):
    max_quant = 2 ** q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    return quantized, scale, min_val


def test_linear_forward_int4():
    input_features = 128  
    out_features = 64  
    batch_size = 32  
    groupsize = 128  
    n_bit = 4  

    inner_k_tiles_options = [2, 4, 8]
    inner_k_tiles = min(inner_k_tiles_options, key=lambda x: abs(x - (input_features // groupsize)))

    x = torch.randn(batch_size, input_features, dtype=torch.bfloat16).cuda()
    
    weight = torch.randn(out_features, input_features, dtype=torch.float16).cuda()

    weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(weight, groupsize, inner_k_tiles)

    output_int4 = linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize).to(torch.float16)

    x_ref = x.to(torch.float16) 
    pseudo_quantized_weight = pseudo_quantize(weight, n_bit)
    output_ref = F.linear(x_ref, pseudo_quantized_weight)
    # output_ref = F.linear(x, pseudo_quantized_weight.to(torch.bfloat16))


    # debug: int32 level
    weight_int32 = group_quantize_tensor(weight, n_bit, groupsize)[0]
    pseudo_quantized_weight_int32 = group_quantize_tensor(weight, n_bit, groupsize)[0]
    print(weight_int32[:5, :5])
    print(pseudo_quantized_weight_int32[:5, :5])
    assert torch.allclose(weight_int32, pseudo_quantized_weight_int32), "The weight_int32 is not close enough to the weight_int4pack!"



    print(output_int4[:5, :5])
    print(output_ref[:5, :5])

    diff = (output_int4 - output_ref).abs()

    print(f'Max difference: {diff.max().item()}')
    print(f'Mean difference: {diff.mean().item()}')
    print(f'Std of difference: {diff.std().item()}')

    assert torch.allclose(output_int4, output_ref, atol=1e-1, rtol=1e-1), "The output of int4 linear forward is not close enough to the reference!"

    print("int4 linear forward test with further validation and comparison passed successfully!")


if __name__ == "__main__":
    test_linear_forward_int4()