import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
# from evaluation.flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
#     replace_mistral_attn_with_flash_attn,
# )
from quest_attention import enable_quest_attention_eval
from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config, change_llama_heavy_const
from h2o_llama import convert_h2o, reset_h2o
from streaming_llama import convert_streaming
from sparq_llama import convert_sparq

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "llama3.1-8b-instruct-128k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-v0.2-hf",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    
    parser.add_argument("--group_factor", type=int, default=2)
    parser.add_argument("--heavy_const", type=int, default=256)
    parser.add_argument("--q_bits", type=int, default=2)
    parser.add_argument("--channel", type=str, default="q")
    
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--ds", action="store_true", help="Enable Double Sparsity Attention")
    parser.add_argument("--h2o", action="store_true", help="Enable H2O Attention")
    parser.add_argument("--streaming", action="store_true", help="Enable StreamingLLM Attention")
    parser.add_argument("--sparq", action="store_true", help="Enable Sparq Attention")

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "llama3.1" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        # clean up memory
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            assert False
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            with torch.no_grad():
                # NOTE(narrativeqa needs to emptify cache with 2 GPUs)
                torch.cuda.empty_cache()
                # NOTE for h2o
                if args.h2o:
                    reset_h2o(model)
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    pred_token_idx = (
                        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    )
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break

            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    elif "llama" in model_name:
        # replace_llama_attn_with_flash_attn()
        print(f"Path: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        # kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "attn_implementation": "flash_attention_2"}
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        # model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(
        #     device
        # )
    elif "longchat" in model_name or "vicuna" in model_name:
        # from fastchat.model import load_model
        # replace_llama_attn_with_flash_attn()
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
    elif "Mistral" in model_name:
        replace_mistral_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model = model.eval()

    if args.quest:
        enable_quest_attention_eval(model, args)
        
    if args.ds:
        # TODO: remove hard-coded path
        channel_path = "/root/DoubleSparse/config/" + path + ".json"
        config = AutoConfig.from_pretrained(path)
        channel_config = None
        with open(channel_path, "r") as f:
            channel_config = json.load(f)
        model = convert_kvcache_llama_heavy_recent(model, config, args.heavy_const, args.group_factor, args.q_bits)
        model = convert_llama_channel_config(model, channel_config, args.channel)

    if args.h2o:
        config = AutoConfig.from_pretrained(path)
        config.cache_budget = args.token_budget
        model = convert_h2o(model, config)
        
    if args.streaming:
        config = AutoConfig.from_pretrained(path)
        model = convert_streaming(model, config, args.token_budget, 8)

    if args.sparq:
        config = AutoConfig.from_pretrained(path)
        model = convert_sparq(model, config, args.token_budget, args.group_factor, args.q_bits)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"pred_e/{model_name}/{dataset}-{args.token_budget}.jsonl"
            elif args.ds:
                out_path = f"pred_e/{model_name}/{dataset}-{args.heavy_const}-{args.group_factor}-{args.q_bits}-{args.channel}.jsonl"
            elif args.h2o:
                out_path = f"pred_e/{model_name}/{dataset}-h2o-{args.token_budget}.jsonl"
            elif args.streaming:
                out_path = f"pred_e/{model_name}/{dataset}-streaming-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.quest:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            elif args.ds:
                out_path = f"pred/{model_name}/{dataset}-{args.heavy_const}-{args.group_factor}-{args.q_bits}-{args.channel}.jsonl"
            elif args.h2o:
                out_path = f"pred/{model_name}/{dataset}-h2o-{args.token_budget}.jsonl"
            elif args.streaming:
                out_path = f"pred/{model_name}/{dataset}-streaming-{args.token_budget}.jsonl"
            elif args.sparq:
                out_path = f"pred/{model_name}/{dataset}-sparq-{args.token_budget}-{args.group_factor}-{args.q_bits}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
