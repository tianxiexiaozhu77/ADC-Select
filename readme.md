1.  **环境准备**

   运行本代码库需要的一些环境，可视情况安装。

   ```
   python==3.8.19
   openai==0.28.0
   vthread==0.1.5
   numpy==1.24.3
   jsonlines==4.0.0 
   argparse==1.4.0
   loguru==0.7.2
   scikit-learn==1.3.2
   ```

   

2. **生成prompt**

​	脚本运行设置：inform/new_data/create_prompt文件夹下，选择目标数据集脚本文件一键运行，以Addsub数据集为例子

```
cd inform
bash new_data/create_promp/Addsub.sh
```

设置openai key，训练集文件地址，用于生成COT的模型类型，输出文件目录，需要挑选出的范例数量等，以Addsub.sh为例，如下



```
eport OPENAI_API_KEY="sk-XXX"					# 设置api-key

export OPENAI_API_BASE='XXX'

DATASET='Addsub'								# 当前任务

TRAIN_PATH='new_data/Addsub/Addsub.jsonl'		# 训练集地址

OUTPUT_DIR='new_data/Addsub/output_new'			# prompt存放位置

MODLE='gpt-3.5-turbo-0613'						# 生成prompt的模型(如果需要生成的话)

EXNUMBER=8										# 得到的prompt的数量

save_file="${OUTPUT_DIR}/inference/${DATASET}_${MODLE}_${TYPE}_0.jsonl"

type_list=(”hard_div”)							# 方法设置，选项为 "COT" "autocot" "complex" "hard_div" 

for TYPE in "${type_list[@]}"

do 

  python code/prompt_generate.py \

  --model $MODLE \

  --dataset $DATASET \

  --data_path $TRAIN_PATH \

  --output_dir $OUTPUT_DIR \

  --example_number $EXNUMBER \

  --type $TYPE 

done
```



3. **预测并评估结果**

inform/script文件夹下，选择目标数据集脚本文件一键运行，以Addsub数据集为例

```
cd inform
bash script/chatgpt_Addsub.sh
```

调整openai key，测试集文件地址，用于生成COT的模型类型，输出文件目录，需要挑选出的范例数量，以Addsub为例，如下

```
export OPENAI_API_KEY="sk-XXX"

export OPENAI_API_BASE='XXX'

type_list=("COT" “autocot” "complex" "hard_div" )

api_name_list=('gpt-3.5-turbo-0613')

for TYPE in "${type_list[@]}"

do

  for api_name in "${api_name_list[@]}"

  do

	 	DATASET='Addsub'
        TEST_PATH='new_data/Addsub/Addsub.jsonl'
        OUTPUT_DIR='new_data/Addsub/output'
        MODLE="chatgpt"
        EXNUMBER=8

        save_file=“YOUR_PATH”


        python -u code/main-p.py \
                --dataset $DATASET \
                --data_path  $TEST_PATH \
                --save_path $OUTPUT_DIR \
                --type $TYPE \
                --example_number $EXNUMBER \
                --save_file $save_file \
                --model $api_name \

        echo "start eval"
        
        python -u code/eval_em_f1.py \
            --dataset $DATASET \
            --data_path $save_file \
            --model $MODLE \
            --type $TYPE \
            > log/${api_name}_${DATASET}_${TYPE}.log 2>&1 & 

    done
done



```













