# source activate ICL_dx
source activate p310
export CUDA_VISIBLE_DEVICES=3
export OPENAI_API_KEY="111"
export OPENAI_API_BASE="https://api.pumpkinaigc.online/query"

# api_name='Llama-3.1-8B-Instruct'
# api_name='gpt-4o-mini'
# api_name='gpt-4-turbo'
api_name='Llama-2-7b-chat-hf'
# api_name='Mistral-7B-Instruct-v0.3'
# api_name='Qwen2.5-Math-7B-Instruct'
# api_name='deepseek-math-7b-rl'
# api_name='Qwen2.5-Math-7B'

TYPE="complex" 
# TYPE="COT"
# TYPE="hard_div"


DATASET='MultiA'
TRAIN_PATH='new_data/MultiA/MultiArith.jsonl'
TEST_PATH='new_data/MultiA/MultiArith.jsonl'
OUTPUT_DIR='new_data/MultiA/zjx'
MODLE="Llama"
PROMPT="Qwen2-Math-7B-ScaleQuest"
# PROMPT="gpt-4-turbo"

EXNUMBER=4
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Meta-Llama-3.1-8B-Instruct'
MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Llama-2-7b-chat-hf'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Mistral-7B-Instruct-v0.3'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B-Instruct'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/deepseek-math-7b-rl'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B'

save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_${TYPE}_cotprompt${PROMPT}.jsonl"


# nohup python -u code/main-p.py \
#     --dataset $DATASET \
#     --data_path  $TEST_PATH \
#     --save_path $OUTPUT_DIR \
#     --type $TYPE \
#     --example_number $EXNUMBER \
#     --save_file $save_file \
#     --api_name $api_name \
#     --api_key_1106 $OPENAI_API_KEY \
#     --model_path $MODEL_PATH \
#     --prompt $PROMPT \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/${api_name}_${DATASET}_${TYPE}_cotprompt${PROMPT}_20241201.log 2>&1 & 

nohup python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
    --type $TYPE \
    > /opt/data/private/zjx/ICL/inform/log/zjx/result/${api_name}_${DATASET}_${TYPE}_20241126.log 2>&1 & 

python -u code/eval_em_f1.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
    --type $TYPE \

# export CUDA_VISIBLE_DEVICES=7
# type_list=(
#     # "COT" 
#     "complex" 
#     # "hard_div" 
#     # "ICL_IE_div" 
#     # "ICL_random"
#     # "ours"
#     )
# api_name_list=('gpt-3.5-turbo-1106')

# for TYPE in "${type_list[@]}"
# do
#     for api_name in "${api_name_list[@]}"
#     do
#         echo "$TYPE $api_name"
#         DATASET='MultiA'
#         TRAIN_PATH='new_data/MultiA/MultiArith.jsonl'
#         TEST_PATH='new_data/MultiA/MultiArith.jsonl'
#         OUTPUT_DIR='new_data/MultiA/zyc'
#         MODLE="chatgpt"
#         EXNUMBER=8

#         save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${api_name}_${TYPE}_0.jsonl"

#         # Question Selection：基于question IE score 挑选example question（hard_div）
#         python code/prompt_generate.py \
#             --dataset $DATASET \
#             --data_path  $TRAIN_PATH \
#             --output_dir $OUTPUT_DIR \
#             --per_sample $TEST_PATH \
#             --example_number $EXNUMBER \
#             --type $TYPE 

#         python -u code/main-p.py \
#             --dataset $DATASET \
#             --data_path  $TEST_PATH \
#             --model $MODLE \
#             --type $TYPE \
#             --example_number $EXNUMBER \
#             --save_file $save_file \
#             --api_name $api_name \

#         python -u code/eval_em_f1.py \
#             --dataset $DATASET \
#             --data_path $save_file \
#             --model $MODLE \
#             --type $TYPE \
#             > log/results/${api_name}_${DATASET}_${TYPE}.log 2>&1 &

#     done
# done