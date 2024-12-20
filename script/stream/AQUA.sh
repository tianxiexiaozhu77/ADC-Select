source activate ICL_dx

export CUDA_VISIBLE_DEVICES=7
api_name='gpt-3.5-turbo-1106'

echo "$api_name"
DATASET='AQuA'
TRAIN_PATH='new_data/AQuA/train.json'
TEST_PATH='new_data/AQuA/test.json'
OUTPUT_DIR='new_data/AQuA/zjx'
API_KEY_1106='sk-YYRIR5TW7Kzq90Bu7351E43fB6Ed43Af9f7dF58a0b8cC0B9'
MODLE=$api_name

EXNUMBER=4
POOL_MAX=1500
THRESHOLD=0.8

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}.jsonl"
echo $save_file

# nohup python code/stream.py \
#     --dataset $DATASET \
#     --api_name $api_name \
#     --api_key_1106 $API_KEY_1106 \
#     --train_path  $TRAIN_PATH \
#     --test_path  $TEST_PATH \
#     --threshold $THRESHOLD \
#     --output_dir $OUTPUT_DIR \
#     --example_number $EXNUMBER \
#     --save_file $save_file \
#     --pool_max $POOL_MAX \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/${DATASET}_${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}_20241118.log 2>&1 &
#     # --pool /opt/data/private/zjx/ICL/inform/new_data/AQuA/zjx/demo_pool/pool_${POOL_MAX}_threshold_${THRESHOLD}.jsonl \


# nohup python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
#     --model $MODLE \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/result/${DATASET}_${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}.log 2>&1 &


python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
    --model $MODLE \


# export CUDA_VISIBLE_DEVICES=7

# api_name='gpt-3.5-turbo-1106'

# echo "$api_name"
# DATASET='AQuA'
# TRAIN_PATH='new_data/AQuA/train.json'
# TEST_PATH='new_data/AQuA/test.json'
# OUTPUT_DIR='new_data/AQuA/zyc'
# MODLE=$api_name

# EXNUMBER=4

# save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_0.jsonl"

# python code/stream.py \
#     --dataset $DATASET \
#     --api_name $api_name \
#     --train_path  $TRAIN_PATH \
#     --test_path  $TEST_PATH \
#     --threshold 0.7 \
#     --output_dir $OUTPUT_DIR \
#     --example_number $EXNUMBER \
#     --save_file $save_file \
#     --pool '/opt/data/private/zyc/ICL/inform/new_data/AQuA/zyc/demo_pool/pool.jsonl' \

# # python -u code/eval_em_f1.py \
# #     --dataset $DATASET \
# #     --data_path $save_file \
# #     --model $MODLE \