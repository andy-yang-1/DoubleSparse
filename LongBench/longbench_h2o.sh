model="llama3.1-8b-instruct-128k"

for task in "gov_report" "triviaqa" "narrativeqa" "qasper" 
do
    python -u pred.py \
        --model $model --task $task

    # for budget in 256 512 1024 2048 4096
    for budget in 256
    do
        CUDA_VISIBLE_DEVICES=2 python -u pred.py \
            --model $model --task $task \
            --h2o --token_budget $budget
    done
done

# python -u eval.py --model $model
