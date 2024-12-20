source activate ICL_dx

export CUDA_VISIBLE_DEVICES=6
# api_name='gpt-4o-mini'
api_name='gpt-4-turbo'

echo "$api_name"
DATASET='CSQA'
TRAIN_PATH='new_data/CSQA/train_rand_split.jsonl'
TEST_PATH='new_data/CSQA/zjx/inference/CSQA_chatgpt_gpt-3.5-turbo-1106_IE_instance.jsonl'
OUTPUT_DIR='new_data/CSQA/zjx'
API_KEY_1106='sk-YYRIR5TW7Kzq90Bu7351E43fB6Ed43Af9f7dF58a0b8cC0B9'
MODLE=$api_name

EXNUMBER=7
POOL_MAX=1500
THRESHOLD=0.8

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_IE_instance.jsonl"
echo $save_file

nohup python code/stream_zjx.py \
    --dataset $DATASET \
    --api_name $api_name \
    --api_key_1106 $API_KEY_1106 \
    --train_path  $TRAIN_PATH \
    --test_path  $TEST_PATH \
    --threshold $THRESHOLD \
    --output_dir $OUTPUT_DIR \
    --example_number $EXNUMBER \
    --save_file $save_file \
    --pool_max $POOL_MAX \
    > /opt/data/private/zjx/ICL/inform/log/zjx/${DATASET}_${api_name}_IE_instance_20241123.log 2>&1 &
    # --pool /opt/data/private/zjx/ICL/inform/new_data/CSQA/zjx/demo_pool/pool_${POOL_MAX}_threshold_${THRESHOLD}.jsonl \


# nohup python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/result/${DATASET}_${api_name}_IE_instance.log 2>&1 &

# python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
