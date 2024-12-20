source activate ICL_dx
export CUDA_VISIBLE_DEVICES=6

# api_name='gpt-4o-mini'
api_name='gpt-4-turbo'

echo "$api_name"
DATASET='GSM8K'
TRAIN_PATH='new_data/GSM8K/train.jsonl'
TEST_PATH='new_data/GSM8K/zjx/inference/GSM8K_chatgpt_gpt-3.5-turbo-1106_IE_instance.jsonl'
OUTPUT_DIR='new_data/GSM8K/zjx'
API_KEY_1106='sk-CLbpLAbwDluA50IjD9D5D39a439c428c9e7eD3D9355b7414'
MODLE=$api_name

EXNUMBER=4
POOL_MAX=1500
THRESHOLD=0.8

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_IE_instance.jsonl"
echo $save_file

# nohup python code/stream_zjx.py \
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
#     > /opt/data/private/zjx/ICL/inform/log/zjx/${DATASET}_${api_name}_IE_instance_20241123.log 2>&1 &
#     # --pool /opt/data/private/zjx/ICL/inform/new_data/GSM8K/zjx/demo_pool/pool_${POOL_MAX}_threshold_${THRESHOLD}.jsonl \


nohup python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
    > /opt/data/private/zjx/ICL/inform/log/zjx/result/${DATASET}_${api_name}_IE_instance.log 2>&1 &

python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
