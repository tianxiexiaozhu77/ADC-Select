export CUDA_VISIBLE_DEVICES=6
source activate p310

# api_name='gpt-3.5-turbo-1106'
api_name='gpt-4o-mini'

echo "$api_name"
DATASET='ASDIV'
TRAIN_PATH='None'
TEST_PATH='new_data/ASDIV/test.jsonl'
OUTPUT_DIR='new_data/ASDIV/zjx'
MODLE=$api_name

EXNUMBER=8
POOL_MAX=1500
THRESHOLD=0.8

# save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_cotprompt${PROMPT}_${EXNUMBER}_${ORDER}_IE_instance.jsonl"
save_file="${OUTPUT_DIR}/inference/${DATASET}_democot${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}_IE_instance.jsonl"
echo $save_file

nohup python code/stream.py \
    --dataset $DATASET \
    --api_name $api_name \
    --api_key_1106 ******** \
    --train_path  $TRAIN_PATH \
    --test_path  $TEST_PATH \
    --threshold $THRESHOLD \
    --output_dir $OUTPUT_DIR \
    --example_number $EXNUMBER \
    --save_file $save_file \
    --pool_max $POOL_MAX \
    > /opt/data/private/zjx/ICL/inform/log/zjx/${DATASET}_democot${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}_20241217.log 2>&1 &
    # --pool /opt/data/private/zyc/ICL/inform/new_data/Addsub/zyc/demo_pool/pool_$POOL_MAX.jsonl \

                                            
# python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path /opt/data/private/zjx/ICL/inform/new_data/Addsub/zjx/inference/Addsub_chatgpt_gpt-3.5-turbo-1106_IE_instance_copy.jsonl \
#     --model $MODLE \

# nohup python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path /opt/data/private/zjx/ICL/inform/new_data/Addsub/zjx/inference/Addsub_gpt-3.5-turbo-1106_pool1500_threshold0.7.jsonl \
#     --model $MODLE \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/result/${DATASET}_${api_name}_pool${POOL_MAX}_threshold${THRESHOLD}.log 2>&1 &


