source activate p310
# DATASET='Addsub'
# DATASET='MultiA'
# DATASET='GSM8K'
# DATASET='MAWPS'
# DATASET='ASDIV'
# DATASET='SVAMP'
DATASET='AQuA'
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/Addsub/Addsub_gpt-4o-mini.jsonl"
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/MultiA/MultiA_gpt-4o-mini.jsonl"
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/GSM8K/GSM8K_gpt-4o-mini.jsonl"
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/MAWPS/MAWPS_gpt-4o-mini.jsonl"
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/ASDIV/ASDIV_gpt-4o-mini.jsonl"
# save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/SVAMP/SVAMP_gpt-4o-mini.jsonl"
save_file="/opt/data/private/zjx/ICL/DAIL/BBH/MATH/new_data/AQuA/DAIL/AQuA_gpt-4o-mini.jsonl"
TYPE="DAIL" 
MODLE="gpt-4o-mini" 


python -u code/eval_em_f1_dail.py \
    --dataset $DATASET \
    --data_path $save_file \
    --model $MODLE \
    --type $TYPE \