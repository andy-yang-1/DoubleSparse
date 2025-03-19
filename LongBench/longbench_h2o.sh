model="llama3.1-8b-instruct-128k"
CUDA_VISIBLE_DEVICES=7

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do

    for budget in 256 1024 4096
    # for budget in 256
    do
        python -u pred.py \
            --model $model --task $task \
            --h2o --token_budget $budget
    done
done

# python -u eval.py --model $model
