export CUDA_VISIBLE_DEVICES=7
type_list=(
    # "COT" 
    "complex" 
    # "hard_div" 
    # "ICL_IE_div" 
    # "ICL_random"
    # "ours"
    )
api_name_list=('gpt-3.5-turbo-1106')

for TYPE in "${type_list[@]}"
do
    for api_name in "${api_name_list[@]}"
    do
        echo "$TYPE $api_name"
        DATASET='MultiA'
        TRAIN_PATH='new_data/MultiA/MultiArith.jsonl'
        TEST_PATH='new_data/MultiA/MultiArith.jsonl'
        OUTPUT_DIR='new_data/MultiA/zyc'
        MODLE="chatgpt"
        EXNUMBER=8

        save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${api_name}_${TYPE}_0.jsonl"

        # Question Selection：基于question IE score 挑选example question（hard_div）
        python code/prompt_generate.py \
            --dataset $DATASET \
            --data_path  $TRAIN_PATH \
            --output_dir $OUTPUT_DIR \
            --per_sample $TEST_PATH \
            --example_number $EXNUMBER \
            --type $TYPE 

        python -u code/main-p.py \
            --dataset $DATASET \
            --data_path  $TEST_PATH \
            --model $MODLE \
            --type $TYPE \
            --example_number $EXNUMBER \
            --save_file $save_file \
            --api_name $api_name \

        python -u code/eval_em_f1.py \
            --dataset $DATASET \
            --data_path $save_file \
            --model $MODLE \
            --type $TYPE \
            > log/results/${api_name}_${DATASET}_${TYPE}.log 2>&1 &

    done
done