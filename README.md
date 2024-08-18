# Double Sparsity

[[paper](https://arxiv.org/pdf/2408.07092)]

![](./assets/double-sparsity-gif-v2.gif)


## TL;DR

We introduce Double Sparsity, a technique to accelerate LLM inference by reducing memory access to the KV cache. 
It predicts important tokens using a subset of channels and computes attention using these important tokens. 
Without any fine-tuning, Double Sparsity achieves bandwidth-efficient attention for the KV cache with almost no loss.


## Install


1. Clone this repo and setup the environment
~~~bash
git clone https://github.com/andy-yang-1/DoubleSparse.git
cd DoubleSparse
conda create -n sparse python=3.9 -y
conda activate sparse
pip install -r requirement.txt
~~~

2. Install torch for offloading feature
~~~bash
# no offloading 
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
~~~

or
~~~bash
# offloading 
pip3 install torch==2.1
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
~~~

## Accuracy Evaluation

1. Wikitext-2 Perplexity
~~~bash
python3 evaluation/perplexity_eval.py --model_path meta-llama/Llama-2-7b-hf --heavy_const 128 --group_factor 4
~~~

2. MMLU
~~~bash
python3 evaluation/mmlu.py
~~~

3. Key-Value Retrieval
~~~bash
python3 evaluation/retrieval_eval.py --model-name-or-path meta-llama/Llama-2-7b-chat-hf
~~~

## Speedup Evaluation

1. Prepare Weight 

~~~bash
cd path/to/DoubleSparse
python3 offloading/scripts/convert_hf_checkpoint.py --checkpoint_dir ~/checkpoints/meta-llama/Llama-2-7b-chat-hf --model_name meta-llama/Llama-2-7b-chat-hf
~~~

2. Attention Operator Speedup

~~~bash
bash scripts/run_attn.sh
~~~

3. End-to-End Inference Speedup

~~~bash
# no offloading
cd models/
python3 generate.py --checkpoint_path path/to/weight/model.pth --max_new_tokens 2048 --batch_size 4
~~~

~~~bash
# offloading
cd offloading/
python3 generate.py --checkpoint_path path/to/weight/model_offloading.pth --max_new_tokens 2048 --batch_size 4
~~~

## Chat with Double Sparsity

Try to chat
~~~
python3 evaluation/chat.py --model_name meta-llama/Llama-2-7b-chat-hf
~~~