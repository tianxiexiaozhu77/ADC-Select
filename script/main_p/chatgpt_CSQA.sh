# source activate ICL_dx
source activate p310
export CUDA_VISIBLE_DEVICES=2
export OPENAI_API_KEY="111"
export OPENAI_API_BASE="https://api.pumpkinaigc.online/query"

# api_name='Llama-3.1-8B-Instruct'
# api_name='gpt-4o-mini'
# api_name='gpt-4-turbo'
# api_name='Llama-2-7b-chat-hf'
# api_name='Mistral-7B-Instruct-v0.3'
# api_name='Qwen2.5-Math-7B-Instruct'
# api_name='deepseek-math-7b-rl'
api_name='Qwen2.5-Math-7B'

TYPE="complex" 
# TYPE="COT"
# TYPE="hard_div"


DATASET='CSQA'
TRAIN_PATH='new_data/CSQA/train_rand_split.jsonl'
TEST_PATH='new_data/CSQA/dev_rand_split.jsonl'
OUTPUT_DIR='new_data/CSQA/zjx'
MODLE="Llama"

EXNUMBER=4
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Meta-Llama-3.1-8B-Instruct'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Llama-2-7b-chat-hf'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Mistral-7B-Instruct-v0.3'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B-Instruct'
# MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/deepseek-math-7b-rl'
MODEL_PATH='/opt/data/private/zjx/ICL/inform_zjx/Qwen2.5-Math-7B'


save_file="${OUTPUT_DIR}/inference/${DATASET}_${api_name}_${TYPE}_0.jsonl"


nohup python -u code/main-p.py \
    --dataset $DATASET \
    --data_path  $TEST_PATH \
    --save_path $OUTPUT_DIR \
    --type $TYPE \
    --example_number $EXNUMBER \
    --save_file $save_file \
    --api_name $api_name \
    --api_key_1106 $OPENAI_API_KEY \
    --model_path $MODEL_PATH \
    > /opt/data/private/zjx/ICL/inform/log/zjx/${api_name}_${DATASET}_${TYPE}_20241129.log 2>&1 & 

# nohup python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
#     --type $TYPE \
#     > /opt/data/private/zjx/ICL/inform/log/zjx/result/${api_name}_${DATASET}_${TYPE}_20241126.log 2>&1 & 

# python -u code/eval_em_f1.py \
#     --dataset $DATASET \
#     --data_path $save_file \
#     --model $MODLE \
#     --type $TYPE \

# # export OPENAI_API_KEY="sk-fvB1sPI0PdSkgVCVC81870F8C35b46A0A029090f7036E2Df"
# # export OPENAI_API_BASE='https://www.gptapi.us/v1'

# type_list=("COT" "complex" "hard_div" "ICL_IE_div" "ICL_random")
# api_name_list=('gpt-3.5-turbo-0613')

# for TYPE in "${type_list[@]}"
# do
#     for api_name in "${api_name_list[@]}"
#     do
#         echo "$TYPE $api_name"

#         DATASET='CSQA'
#         TRAIN_PATH='new_data/CSQA/train_rand_split.jsonl'
#         TEST_PATH='new_data/CSQA/dev_rand_split.jsonl'
#         OUTPUT_DIR='new_data/CSQA/output_ours'
        
#         MODLE='chatgpt'
#         EXNUMBER=7
#         save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${api_name}_${TYPE}_0.jsonl"

#         # python code/prompt_generate.py \
#         #     --dataset $DATASET \
#         #     --data_path  $TRAIN_PATH \
#         #     --output_dir $OUTPUT_DIR \
#         #     --example_number $EXNUMBER \
#         #     --type $TYPE 

#         # python -u code/main-p.py \
#         #     --dataset $DATASET \
#         #     --data_path  $TEST_PATH \
#         #     --model $MODLE \
#         #     --type $TYPE \
#         #     --example_number $EXNUMBER \
#         #     --save_file $save_file \
#         #     --api_name $api_name \

#         python -u code/eval_em_f1.py \
#             --dataset $DATASET \
#             --data_path $save_file \
#             --model $MODLE \
#             --type $TYPE \
#             # > log/${MODLE}_${DATASET}_${TYPE}.log 2>&1 & 
#     done
# done