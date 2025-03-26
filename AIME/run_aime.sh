CUDA_VISIBLE_DEVICES=$1
budget=$2

model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
architecture="qwen2"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 aime.py --model_path $model_path \
    --architecture $architecture --mode ds \
    --heavy_const $budget \
    --output_path "output/aime24_ds_${budget}.jsonl"

