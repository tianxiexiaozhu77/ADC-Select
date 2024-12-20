export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export OPENAI_API_BASE='YOUR_OPENAI_KEY_BASE'

DATASET='MultiA'
TRAIN_PATH='new_data/MultiA/MultiArith.jsonl'
TEST_PATH='new_data/MultiA/MultiArith.jsonl'
OUTPUT_DIR='new_data/MultiA/output_new'

MODLE='gpt-3.5-turbo-0613'
EXNUMBER=8

save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${TYPE}_0.jsonl"


type_list=("complex" "hard_div")
for TYPE in "${type_list[@]}"
do 
    python code/prompt_generate.py \
    --model $MODLE \
    --dataset $DATASET \
    --data_path  $TRAIN_PATH \
    --output_dir $OUTPUT_DIR \
    --example_number $EXNUMBER \
    --type $TYPE 
done