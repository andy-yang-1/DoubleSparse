model="llama3.1-8b-instruct-128k"

for task in "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa" "narrativeqa" "qasper" 
do
    python -u pred.py \
        --model $model --task $task

    # for budget in 256 512 1024 2048 4096
    for budget in 4096
    do
        CUDA_VISIBLE_DEVICES=7 python -u pred.py \
            --model $model --task $task \
            --streaming --token_budget $budget
    done
done

# python -u eval.py --model $model
