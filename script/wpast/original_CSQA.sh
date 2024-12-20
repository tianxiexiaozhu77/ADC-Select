# export OPENAI_API_KEY="sk-fvB1sPI0PdSkgVCVC81870F8C35b46A0A029090f7036E2Df"
# export OPENAI_API_BASE='https://www.gptapi.us/v1'

type_list=("COT" "complex" "hard_div" "ICL_IE_div" "ICL_random")
api_name_list=('gpt-3.5-turbo-0613')

for TYPE in "${type_list[@]}"
do
    for api_name in "${api_name_list[@]}"
    do
        echo "$TYPE $api_name"

        DATASET='CSQA'
        TRAIN_PATH='new_data/CSQA/train_rand_split.jsonl'
        TEST_PATH='new_data/CSQA/dev_rand_split.jsonl'
        OUTPUT_DIR='new_data/CSQA/output_ours'
        
        MODLE='chatgpt'
        EXNUMBER=7
        save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${api_name}_${TYPE}_0.jsonl"

        # python code/prompt_generate.py \
        #     --dataset $DATASET \
        #     --data_path  $TRAIN_PATH \
        #     --output_dir $OUTPUT_DIR \
        #     --example_number $EXNUMBER \
        #     --type $TYPE 

        # python -u code/main-p.py \
        #     --dataset $DATASET \
        #     --data_path  $TEST_PATH \
        #     --model $MODLE \
        #     --type $TYPE \
        #     --example_number $EXNUMBER \
        #     --save_file $save_file \
        #     --api_name $api_name \

        python -u code/eval_em_f1.py \
            --dataset $DATASET \
            --data_path $save_file \
            --model $MODLE \
            --type $TYPE \
            # > log/${MODLE}_${DATASET}_${TYPE}.log 2>&1 & 
    done
done