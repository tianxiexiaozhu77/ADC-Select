import os
import openai
import torch
import random
import json
from transformers import GenerationConfig  
from api_service_zjx import api_get_tokens
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 从文件系统加载特定数据集的提示文本
def get_prompt(args, prompt):  # 对一个包含多个提示文本的字符串进行重新排序，并返回处理后的提示文本
    prompt_list = prompt.rstrip('\n\n\n').split('\n\n\n')

    new_list = [None] * len(prompt_list)
    left_index = 0
    right_index = len(prompt_list) - 1
    for i, value in enumerate(prompt_list):
        if i % 2 == 0:  # 如果 i 是偶数，将 value 放在 new_list 的末尾位置（right_index）并递减 right_index
            new_list[right_index] = value
            right_index -= 1
        else:  # 如果 i 是奇数，将 value 放在 new_list 的开头位置（left_index）并递增 left_index
            new_list[left_index] = value
            left_index += 1

    # prompt=prompt.split('\n\n\n')
    prompt="\n\n".join(new_list)+"\n\n"
    return prompt

# 封装模型的加载、配置和生成逻辑
class Inference(object):
    def __init__(self, model=None, device=0, generate=False, example=None):
        self.model = model 
        self.bot = None  
        self.hug_classifier = None  
        self.device = device  
        self.examples = example 

    # 根据不同模型与数据集动态构建AI模型输入提示，并添加问题及引导信息
    def generate_input(self, question, option_str, args, prompt): 
        fin_input = "" 
        if args.dataset in ['GSM8K', 'MultiA', 'Strategyqa', 'date_understanding', 'Addsub']:
            if args.type not in ["COT"]:
                fin_input = get_prompt(args, prompt)

            if args.dataset == 'date_understanding':
                option_str1 = option_str.replace(") ",")").replace("  ("," (").strip()
                fin_input = fin_input + "Question: " + question + "\n"+ option_str1 + "\nAnswer: Let's think step by step.\n" # 没有Option: ?
                
            else:
                fin_input = fin_input + "Question: " + question + "\nAnswer: Let's think step by step.\n" 
                
        elif args.dataset in ['AQuA']:
            if args.type not in ["COT"]:
                fin_input = get_prompt(args, prompt)
            
            fin_input = fin_input + "Question: " + question + "\nOptions: "+ option_str + "\nAnswer: Let's think step by step.\n"  
        
        elif args.dataset in ['CSQA']:
            if args.type not in ["COT"]:
                fin_input = get_prompt(args, prompt)
                option_str1 = option_str.replace(") ",")").replace("  ("," (").strip()
            fin_input = fin_input + "\n Question: " + question + "\n" + option_str1 + "\nAnswer: Let's think step by step.\n" 

        return fin_input

    def predict_with_api_service(self, question, option, args,res_dict,f,logger,prompt,model,tokenizer):
        global api_count
        # input = self.generate_input(question, option, args, prompt)
        # res_dict["prompt"]=input
        api_get_tokens(args.api_name,res_dict["prompt"],res_dict,f,logger,args,model,tokenizer)



