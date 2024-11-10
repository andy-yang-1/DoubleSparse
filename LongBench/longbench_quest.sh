model="llama3.1-8b-instruct-128k"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do
    python -u pred.py \
        --model $model --task $task

    for budget in 256 512 1024 2048 4096
    do
        CUDA_VISIBLE_DEVICES=2 python -u pred.py \
            --model $model --task $task \
            --quest --token_budget $budget --chunk_size 16
    done
done

# python -u eval.py --model $model
