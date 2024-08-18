import argparse
import os
import re
import json
from tqdm import tqdm
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np

from fastchat.model import load_model, get_conversation_template
# from utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample 

from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from streaming_llama import convert_streaming, change_streaming_para
from rtn_llama import convert_rtn, change_rtn_para
# from h2o_llama import convert_h2o, reset_h2o
from advanced_h2o_llama import convert_h2o, reset_h2o
from sparq_llama import convert_sparq
from get_config import get_best_config
from offload_llama import convert_kvcache_llama_offloading, convert_llama_offloading_channel_config


def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)
    return test_cases

def test_lines_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]

    if "longchat" in args.model_name_or_path:
        conv = get_conversation_template("vicuna")
    else:
        conv = get_conversation_template(args.model_name_or_path)
    print(f"Using conversation template: {conv.name}")
    if "mosaicml/mpt-30b-chat" in args.model_name_or_path:
        prompt += f'Answer in the format <{test_case["random_idx"][0]}> <REGISTER_CONTENT>.'
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]
    
    # Disable use_cache if using longchat models with flash attention
    use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)
    device = getattr(model, "device", "cpu")
    
    output = model.generate(input.input_ids.to(device), max_new_tokens=100, use_cache=use_cache)[0]
    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)
    # if idx ==0:
    #     with open(output_file, "w") as f:
    #         f.write(summary)
    #         f.write("\n")
    # else:
    #     with open(output_file, "a+") as f:
    #         f.write(summary)
    #         f.write("\n")
    
    return expected_number == response_number, prompt_length, summary

def longeval_test(model, tokenizer, output_dir, args):
            
    if args.task == "lines":
        # for num_lines in [50, 100, 200]:
        for num_lines in [50]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
            
            # output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=None, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
                reset_h2o(model)
            accuracy = num_correct / len(test_cases)

            # with open(output_file, "a+") as f:
            #     f.write(f"Accuracy: {accuracy}")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
            if args.eval_shortest_only:
                break
    else:
        print(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="model path")
    parser.add_argument("--task", type=str, default="lines", help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_gpu_memory", type=int, default=40, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--test_dir", type=str, default="./", help="Directory of the testcases")
    parser.add_argument("--framework", type=str, default=None, help="Framework for serving")
    args = parser.parse_args()

    # maybe_monkey_patch(args)
    output_dir = None

    # h2o
    # model_path = args.model_name_or_path
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    # config.heavy_ratio = 0.04
    # config.recent_ratio = 0.0225
    # model = convert_h2o(model, config)
    # longeval_test(model, tokenizer, output_dir, args)

    # streaming llm
    # model_path = args.model_name_or_path
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    # model = convert_streaming(model, config, 128, 4)
    # local_consts = [1280 // factor for factor in [1, 2, 4, 8, 16, 32]]
    # for local_const in local_consts:
    #     model = change_streaming_para(model, local_const-4, 4)
    #     # model = change_streaming_para(model, local_const, 0)
    #     longeval_test(model, tokenizer, output_dir, args)

    
    # rtn
    # model_path = args.model_name_or_path
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    # model = convert_rtn(model, config, 16)
    # qbits = [3]
    # for qbit in qbits:
    #     model = change_rtn_para(model, qbit)
    #     # model = change_streaming_para(model, local_const, 0)
    #     longeval_test(model, tokenizer, output_dir, args)


    # sparq
    # model_path = args.model_name_or_path
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    # sparsity_factor = 24
    # model = convert_sparq(model, config, 1280 // sparsity_factor, 128 // sparsity_factor)
    # longeval_test(model, tokenizer, output_dir, args)


    # double sparsity

    # model_path = args.model_name_or_path
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    # sparsity_factor = 16
    # model = convert_kvcache_llama_heavy_recent(model, config, 1280 // sparsity_factor, 2, 4)
    # channel_path = "llama2-7b-chat-qk-channel-config.json"
    # with open(channel_path, "r") as f:
    #     channel_config = json.load(f)
    # model = convert_llama_channel_config(model, channel_config, "qk")
    # longeval_test(model, tokenizer, output_dir, args)


    # double sparsity offloading

    model_path = args.model_name_or_path
    device = "cuda"
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    sparsity_factor = 32
    model = convert_kvcache_llama_offloading(model, config, 1280 // sparsity_factor, 2, 1, [0,1,31])
    channel_path = "llama2-7b-chat-qk-channel-config.json"
    with open(channel_path, "r") as f:
        channel_config = json.load(f)
    model = convert_llama_offloading_channel_config(model, channel_config, "qk")
    longeval_test(model, tokenizer, output_dir, args)


    # best_configs = get_best_config()

    # for model_path, heavy_config in best_configs.items():
    #     if "chat" not in model_path and "vicuna" not in model_path:
    #         continue
    #     args.model_name_or_path = model_path
    #     model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    #     tokenizer = LlamaTokenizer.from_pretrained(model_path)
    #     config = AutoConfig.from_pretrained(model_path)

    #     channel_path = heavy_config["channel_path"]

    #     sparsity_factor = heavy_config["sparsity_factor"]
    #     qbit = heavy_config["qbit"]
    #     # sparsity_factor = 24
    #     # qbit = 3
    #     group_factor = sparsity_factor * qbit // 16
    #     channel_type = heavy_config["channel_type"]

    #     channel_config = None
    #     with open(channel_path, "r") as f:
    #         channel_config = json.load(f)

    #     model = convert_kvcache_llama_heavy_recent(model, config, 1280 // sparsity_factor, group_factor, qbit)
    #     model = convert_llama_channel_config(model, channel_config, channel_type)

    #     longeval_test(model, tokenizer, output_dir, args)
    #     model = None
    #     gc.collect()
    #     torch.cuda.empty_cache()
