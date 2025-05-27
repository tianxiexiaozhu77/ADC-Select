import torch
from dataload import Dataset
import jsonlines  
import json,re
from tqdm import tqdm
import numpy as np
import os
from openai import OpenAI
import argparse,random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models, util
import random
from pool import Pool
from inference_all import Inference
from loguru import logger
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

client = OpenAI(
        base_url="https://www.gptapi.us/v1",
        api_key="********"
    )

def get_Cot_generation_prompt(dataset_name,question,option,answer):
    
    Cot_generation_prompt = {
    "CSQA" : str(
    "Q: " + question + "\n" +str(option) + "\n" +f"A: {str(answer)}\n"+ "Let's think step by step. " 
    "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    ),
    "date_understanding" : str(
    "Q: " + question + "\n" + "Option:"+str(option) + "\n" + f"A: {str(answer)}\n" + "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E/F).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    ),
    "Addsub" : str(
    "Q: " + question + "\n" + "Let's think step by step. "  # "Q: " + question + "\n" + f"A: {str(answer)}\n" + "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "MultiA" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" +  "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "GSM8K" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" +  "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "Strategyqa" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" +  "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer in (Yes or No) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer (Yes or No) is Yes.\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer (Yes or No) is No.\"}}\n"
    "Output: "
    ),
    "AQuA" : str(
    "Q: " + question + "\n" +str(option) + "\n" + "Let's think step by step. "  # "Q: " + question + "\n" + "Option:"+str(option) + "\n" +f"A: {str(answer)}\n"+ "Let's think step by step. "
    "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    )
    }
    return Cot_generation_prompt[dataset_name]

def cot_sc(args, questions, options, model,tokenizer, times=5):
    '''
    record中多选会多记录一条option
    因为生成的数据答案不可信，所以不设标答
    '''
    # 获取CoT然后计算每个CoT的IE（重复多次）
    record_path = os.path.join(args.output_dir, f"cot_generation/{args.dataset}_{args.api_name}_cot_record_{args.pool_max}_threshold_{args.threshold}.jsonl")  # {args.dataset}_{args.api_name}_cot_record({args.pool_max})_threshold_{args.threshold}.jsonl
    record_question = []
    
    if os.path.exists(record_path):
        record = []
        with open(record_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                record.append(data)
        record_question = [i['question'] for i in record]

    
    for cnt in range(times):
        target_path = os.path.join(args.output_dir,f"cot_generation/pool.jsonl")  # cot_generation/{args.dataset}_{args.api_name}_{args.type}_{cnt}_pool_{args.pool_max}_threshold_{args.threshold}.jsonl
        with open(target_path, "w", encoding='utf-8') as f:
            for i in tqdm(range(len(questions))):
                if questions[i] in record_question:
                    # 如果池子里有已经生成的CoT就跳过生成，直接检索然后整个条目添加进jsonl中
                    f.write(json.dumps(record[record_question.index(questions[i])]) + '\n')  # 找到某个demo对应的cot
                    f.flush()
                    continue    
                new_dict = {}
                if args.dataset in ['CSQA','date_understanding', 'AQuA']:
                    # 多选
                    option_str = ' '.join(['('+i for i in options[i]])
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], option_str, None)
                    new_dict["option"] = option_str                    
                else: 
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], None, None)
                new_dict["question"] = questions[i]
                # 获取COT
                while True:
                    try:
                        response = client.chat.completions.create(
                            model='gpt-4o-mini',
                            messages=[
                                {
                                    "role": "user",
                                    "content": fin
                                }
                            ],
                            temperature=0.7,
                            top_p=0.9,
                            frequency_penalty=0.5,
                            presence_penalty=0.5
                        )
                        cleaned_res = response.choices[0].message.content.replace('```json\n', '').replace('\n```', '').replace('\n',' ')
                        res = eval(cleaned_res)
                        # break
                        new_dict["cot"] = str(res["Reasoning"])
                        new_dict["answer"]=res["Answer"]
                        break
                    except Exception as e:
                        print(f"fails: {e}")
                        continue
                new_dict["score"] = 0
                f.write(json.dumps(new_dict) + '\n')
                f.flush()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_name', type=str, default='gpt-3.5-turbo-1106',
                        choices=['gpt-3.5-turbo-1106', 'Llama-3.1-8B-Instruct'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--api_key_1106', type=str)
    parser.add_argument("--dataset", type=str, default="CSQA",
                        choices=["AQuA", "GSM8K", "MultiA", "Addsub", "MathQA", "CSQA", "Strategyqa", "date_understanding"])
    parser.add_argument('--train_path', type=str, default='None')
    parser.add_argument('--test_path', type=str, default='data/CSQA/train_rand_split.jsonl')
    parser.add_argument('--output_dir', type=str, default='data/CSQA/output')
    parser.add_argument('--type', type=str, default="all")
    parser.add_argument('--example_number', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_file", type=str)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--pool_max", type=int, default=1000)
    args = parser.parse_args()  
    return args 

if __name__ == '__main__':
    args = get_args()
    options, model,tokenizer = None, None, None
    
    questions_list = []
    r_path = '/opt/data/private/zjx/ICL/inform/new_data/Addsub/zjx/cot_generation/Addsub_gpt-3.5-turbo-1106_1500_threshold_0.9_questions_empty.jsonl'
    with open(r_path, 'r', encoding='utf-8') as f:
        for data in jsonlines.Reader(f):
            questions = []
            for question in data["questions"].values():
                questions.append(question)
            questions_list.append(questions)
            
    print(1)
    
    for questions in questions_list:
        cot_sc(args, questions, options, model,tokenizer,times=1)
