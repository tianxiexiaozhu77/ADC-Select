source activate ICL_dx
# source activate p310
export CUDA_VISIBLE_DEVICES=5

api_name='Llama-3.1-8B-Instruct'
# api_name='Llama-2-7b-chat-hf'
# api_name='Mistral-7B-Instruct-v0.3'
# api_name='Qwen2.5-Math-7B-Instruct'
# api_name='deepseek-math-7b-rl'
# api_name='gpt-4-turbo'

echo "$api_name"
DATASET='GSM8K'
TRAIN_PATH='new_data/GSM8K/train.jsonl'
TEST_PATH='new_data/GSM8K/test.jsonl'
OUTPUT_DIR='new_data/GSM8K/zjx'
API_KEY_1106='sk-CLbpLAbwDluA50IjD9D5D39a439c428c9e7eD3D9355b7414'
MODLE=$api_name

EXNUMBER=4
POOL_MAX=1500
THRESHOLD=0.8

# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Llama-2-7b-chat-hf'
MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Mistral-7B-Instruct-v0.3'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B-Instruct'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/deepseek-math-7b-rl'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B'
# '/opt/data/private/zjx/ICL/inform_zjx/Meta-Llama-3.1-8B-Instruct'
# /opt/data/private/zjx/ICL/inform_zjx/Llama-2-7b-chat-hf

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_IE_instance.jsonl"
echo $save_file

nohup python code/stream.py \
    --dataset $DATASET \
    --api_name $api_name \
    --api_key_1106 $API_KEY_1106 \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --threshold $THRESHOLD \
    --output_dir $OUTPUT_DIR \
    --example_number $EXNUMBER \
    --save_file $save_file \
    --pool_max $POOL_MAX \
    --model_path $MODEL_PATH \
    > /opt/data/private/zjx/ICL/inform/log/zjx/${DATASET}_${api_name}_IE_instance_20241130.log 2>&1 &
# # # #     # --pool /opt/data/private/zjx/ICL/inform/new_data/GSM8K/zjx/demo_pool/pool_${POOL_MAX}_threshold_${THRESHOLD}.jsonl \


# python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \

# nohup python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/result/${DATASET}_${api_name}_IE_instance_20241126.log 2>&1 &

