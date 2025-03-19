model="llama3.1-8b-instruct-128k"
CUDA_VISIBLE_DEVICES=4

# for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
for task in "gov_report" "triviaqa"
do

    for budget in 256 1024 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --ds --heavy_const $budget --group_factor 2 --q_bits 2 --channel q
    done
done

# python -u eval.py --model $model
