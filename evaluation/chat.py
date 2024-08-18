import argparse
import os
import re
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np

from fastchat.model import get_conversation_template
from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from modify_mistral import convert_kvcache_mistral_heavy_recent, convert_mistral_channel_config
# from modify_mixtral import convert_kvcache_mixtral_heavy_recent, convert_mixtral_channel_config
from streaming_llama import convert_streaming
from rtn_llama import convert_rtn
from offload_llama import convert_kvcache_llama_offloading, convert_llama_offloading_channel_config
from offload_mistral import convert_kvcache_mistral_offloading, convert_mistral_offloading_channel_config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Selected model')
    parser.add_argument('--offloading', action='store_true', help='Whether to use offloading')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral"])
    parser.add_argument('--channel', type=str, default="qk", choices=["q", "k", "qk"])
    parser.add_argument('--heavy_const', type=int, default=128, help='Heavy constant')
    parser.add_argument('--group_factor', type=int, default=4, help='Group factor')
    parser.add_argument('--q_bits', type=int, default=4, help='Quantization bits')

    args = parser.parse_args()


    model_path = args.model_path
    channel_path = "config/" + model_path + ".json"


    if "70b" in model_path:
        # TODO: support more than 8 x a10g
        device_map = {"model.embed_tokens": 0, "model.norm": 7, "lm_head": 7}
        for i in range(80):
            device_map[f"model.layers.{i}"] = i // 10
    else:
        device_map = "auto"

    kwargs = {"torch_dtype": torch.float16}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).cuda()
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

    conv = get_conversation_template(model_path)

    while True:
        print(f"{conv.roles[0]}:", end="")
        inp = input()
        if inp == "quit":
            break
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        prompt_length = input_ids.shape[-1]
        output = model.generate(input_ids, do_sample=True, max_new_tokens=2048-prompt_length, use_cache=True)[0]
        
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

        print(f"{conv.roles[1]}:{output}")
        conv.update_last_message(output)
