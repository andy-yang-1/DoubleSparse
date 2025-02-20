import json
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from datasets import load_dataset
from functools import partial
import gc

from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from modify_mistral import convert_kvcache_mistral_heavy_recent, convert_mistral_channel_config
from modify_qwen2 import convert_kvcache_qwen2_heavy_recent, convert_qwen2_channel_config
# from modify_mixtral import convert_kvcache_mixtral_heavy_recent, convert_mixtral_channel_config
from streaming_llama import convert_streaming
from rtn_llama import convert_rtn
from offload_llama import convert_kvcache_llama_offloading, convert_llama_offloading_channel_config
from offload_mistral import convert_kvcache_mistral_offloading, convert_mistral_offloading_channel_config


def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    torch.cuda.empty_cache()
    gc.collect()

    max_seq_len = 4096

    testenc = testenc.input_ids.to(model.device)
    print(testenc.shape)
    nsamples = testenc.shape[1] // max_seq_len
    model = model.eval()

    nlls = []

    # 57 -> nan

    gc.collect()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * max_seq_len):((i + 1) * max_seq_len)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * max_seq_len):((i + 1) * max_seq_len)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_seq_len
        nlls.append(neg_log_likelihood)
        batch = None
        torch.cuda.empty_cache()
        gc.collect()

    return torch.exp(torch.stack(nlls).sum() / (nsamples * max_seq_len))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Selected model')
    parser.add_argument('--offloading', action='store_true', help='Whether to use offloading')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral", "qwen2"])
    parser.add_argument('--channel', type=str, default="qk", choices=["q", "k", "qk"])
    parser.add_argument('--heavy_const', type=int, default=128, help='Heavy constant')
    parser.add_argument('--group_factor', type=int, default=2, help='Group factor')
    parser.add_argument('--q_bits', type=int, default=2, help='Quantization bits')

    args = parser.parse_args()


    model_path = args.model_path
    channel_path = "config/" + model_path + ".json"


    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)


    channel_config = None
    with open(channel_path, "r") as f:
        channel_config = json.load(f)

    if args.offloading:
        if args.architecture == "llama":
            model = convert_kvcache_llama_offloading(model, config, args.heavy_const, args.group_factor, args.q_bits, [0,1,31])
            model = convert_llama_offloading_channel_config(model, channel_config, args.channel)
        elif args.architecture == "mistral":
            model = convert_kvcache_mistral_offloading(model, config, args.heavy_const, args.group_factor, args.q_bits, [0,1,31])
            model = convert_mistral_offloading_channel_config(model, channel_config, args.channel)
    else:
        if args.architecture == "llama":
            model = convert_kvcache_llama_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_llama_channel_config(model, channel_config, args.channel)
        elif args.architecture == "mistral":
            model = convert_kvcache_mistral_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_mistral_channel_config(model, channel_config, args.channel)
        elif args.architecture == "qwen2":
            model = convert_kvcache_qwen2_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_qwen2_channel_config(model, channel_config, args.channel)


    # model = convert_kvcache_mixtral_heavy_recent(model, config, 128, 4, 4)
    # model = convert_mixtral_channel_config(model, channel_config, "q") #TODO: no k outlier for gqa

    # model = convert_streaming(model, config, 128, 4)
    # model = convert_rtn(model, config, 2)

    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    print(evaluate(model, tokenizer))


