import argparse
import os
import re
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np

# from fastchat.model import get_conversation_template
from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from modify_qwen2 import convert_kvcache_qwen2_heavy_recent, convert_qwen2_channel_config



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.1-8B", help='Selected model')
    parser.add_argument('--mode', type=str, default="ds", choices=["ds", "ds-offload", "dense"], help='Whether to use offloading')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral", "qwen2"])
    parser.add_argument('--channel', type=str, default="q", choices=["q", "k", "qk"])
    parser.add_argument('--heavy_const', type=int, default=128, help='Heavy constant')
    parser.add_argument('--group_factor', type=int, default=2, help='Group factor')
    parser.add_argument('--q_bits', type=int, default=2, help='Quantization bits')
    parser.add_argument('--output_path', type=str, required=True, help='Prompt for generation')

    args = parser.parse_args()


    model_path = args.model_path
    channel_path = "../config/" + model_path + ".json"


    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}


    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)


    channel_config = None
    with open(channel_path, "r") as f:
        channel_config = json.load(f)


    if args.mode == "ds":
        if args.architecture == "llama":
            model = convert_kvcache_llama_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_llama_channel_config(model, channel_config, args.channel)
        elif args.architecture == "mistral":
            model = convert_kvcache_mistral_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_mistral_channel_config(model, channel_config, args.channel)
        elif args.architecture == "qwen2":
            model = convert_kvcache_qwen2_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
            model = convert_qwen2_channel_config(model, channel_config, args.channel)
    elif args.mode == "ds-offload":
        if args.architecture == "llama":
            model = convert_kvcache_llama_offloading(model, config, args.heavy_const, args.group_factor, args.q_bits, [0,1,31])
            model = convert_llama_offloading_channel_config(model, channel_config, args.channel)
        elif args.architecture == "mistral":
            model = convert_kvcache_mistral_offloading(model, config, args.heavy_const, args.group_factor, args.q_bits, [0,1,31])
            model = convert_mistral_offloading_channel_config(model, channel_config, args.channel)

    
    data_path = "./data/aime24.jsonl"
    print("Reading data from: ", data_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    print("Save output to: ", args.output_path)
    with open(args.output_path, 'w', encoding='utf-8') as g:
        for item in tqdm(data):
            prompt = item['prompt']
            answer = item['answer']
            question = "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n"

            input_ids = tokenizer(question, return_tensors="pt").input_ids.cuda()
            max_new_tokens = 32000-input_ids.shape[-1]

            output = model.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)[0][len(input_ids[0]):]
            output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]
            item['gen'] = output
            print(output)
            g.write(json.dumps(item, ensure_ascii=False) + '\n')
            g.flush()

    # prompt = args.prompt
    # prompt = "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    # max_new_tokens = 8000-input_ids.shape[-1]

    # output = model.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)[0]
    # output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]
    # print(output)


    # prompt = "<｜begin▁of▁sentence｜><｜User｜>Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\nPlease reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n"

