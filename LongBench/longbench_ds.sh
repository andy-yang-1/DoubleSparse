model="llama3.1-8b-instruct-128k"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do

    for budget in 256 512 1024 2048 4096
    do
        CUDA_VISIBLE_DEVICES=3 python -u pred.py \
            --model $model --task $task \
            --ds --heavy_const $budget --group_factor 2 --q_bits 2 --channel q
    done
done

# python -u eval.py --model $model
