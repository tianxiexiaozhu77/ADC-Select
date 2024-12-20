export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export OPENAI_API_BASE='YOUR_OPENAI_KEY_BASE'

DATASET='AQuA'
TRAIN_PATH='new_data/AQuA/train.json'
TEST_PATH='new_data/AQuA/test.json'
OUTPUT_DIR='new_data/AQuA/output_7_0.9'


MODLE='gpt-3.5-turbo-0613' 
EXNUMBER=7
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

