export CUDA_VISIBLE_DEVICES=7

api_name='gpt-3.5-turbo-1106'

echo "$api_name"
DATASET='Strategyqa'
TRAIN_PATH='None'
TEST_PATH='new_data/Strategyqa/task.json'
OUTPUT_DIR='new_data/Strategyqa/zyc'
MODLE=$api_name

EXNUMBER=6

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_0.jsonl"

# python code/stream.py \
#     --dataset $DATASET \
#     --train_path $TRAIN_PATH \
#     --api_name $api_name \
#     --test_path $TEST_PATH \
#     --threshold 0.6 \
#     --output_dir $OUTPUT_DIR \
#     --example_number $EXNUMBER \
#     --save_file $save_file \
#     --pool '/opt/data/private/zyc/ICL/inform/new_data/Strategyqa/zyc/demo_pool/pool.jsonl' \
# save_file='/opt/data/private/zyc/ICL/inform/new_data/Strategyqa/zyc/inference/Strategyqa_gpt-3.5-turbo-1106_0.jsonl'
python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \