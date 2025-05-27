import torch
from dataload_zjx import Dataset
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
from inference_all_zjx import Inference
from loguru import logger
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

random.seed(42)

client = OpenAI(
        base_url="https://www.gptapi.us/v1",
        api_key="********"
    )

# embedding model
model_path = '/opt/data/private/zyc/Models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
EMBEDDING = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# COT 生成prompt templete
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
    "Please output a concise and correct reasoning first and then output the answer in (arabic numerals) format.\n"  # "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
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
    "Please output a concise and correct reasoning first and then output the answer from provided options(A/B/C/D/E).\n"  # "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    )
    }
    return Cot_generation_prompt[dataset_name]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_name', type=str, default='gpt-3.5-turbo-1106',
                        choices=['gpt-3.5-turbo-1106', 'Llama-3.1-8B-Instruct','gpt-4o-mini','gpt-4-turbo','Llama-2-7b-chat-hf','Mistral-7B-Instruct-v0.3','Qwen2.5-Math-7B-Instruct','deepseek-math-7b-rl'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--api_key_1106', type=str)
    parser.add_argument("--dataset", type=str, default="CSQA",
                        choices=["AQuA", "GSM8K", "MultiA", "Addsub", "MathQA", "CSQA", "Strategyqa", "date_understanding","MAWPS","ASDIV","SVAMP"])
    parser.add_argument('--train_path', type=str, default='None')
    parser.add_argument('--test_path', type=str, default='data/CSQA/train_rand_split.jsonl')
    parser.add_argument('--output_dir', type=str, default='data/CSQA/output')
    parser.add_argument('--type', type=str, default="all")
    parser.add_argument('--example_number', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_file", type=str)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--pool_max", type=int, default=1000)
    parser.add_argument('--order', type=str)
    args = parser.parse_args()  
    return args  

def flexible_match(prediction,ground_truth):
    if prediction is None or ground_truth is None:
        return 0
    
    numeric_re = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
    if bool(numeric_re.match(prediction)) and bool(numeric_re.match(ground_truth)):
        if abs(float(prediction)-float(ground_truth)) < 0.0001:
            return 1
        else :
            return 0
    else:
        return int(prediction.strip().lower() == ground_truth.strip().lower())


def cot_sc(args, questions, options, model,tokenizer, times=5):
    '''
    record中多选会多记录一条option
    因为生成的数据答案不可信，所以不设标答
    '''
    # 获取CoT然后计算每个CoT的IE（重复多次）
    record_path = os.path.join(args.output_dir, f"cot_generation/{args.dataset}_{args.api_name}_cot_record({args.pool_max})_threshold_{args.threshold}.jsonl")
    record_question = []
    
    if os.path.exists(record_path):
        record = []
        with open(record_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                record.append(data)
        record_question = [i['question'] for i in record]

    
    for cnt in range(times):
        target_path = os.path.join(args.output_dir,f"cot_generation/{args.dataset}_{args.api_name}_{args.type}_{cnt}_pool_{args.pool_max}_threshold_{args.threshold}.jsonl")

        with open(target_path, "w", encoding='utf-8') as f:
            for i in tqdm(range(len(questions))):
                if questions[i] in record_question:
                    # 如果池子里有已经生成的CoT就跳过生成，直接检索然后整个条目添加进jsonl中
                    f.write(json.dumps(record[record_question.index(questions[i])]) + '\n')
                    continue
                    
                new_dict = {}

                if args.dataset in ['AQuA']:  # 
                    # 多选
                    if 'gpt-3.5-turbo-1106' in args.api_name:
                        option_str = ' '.join(['('+i for i in options[i]])
                        # fin = get_Cot_generation_prompt(args.dataset, questions[i], option_str, None)
                    if 'Llama' in args.api_name:
                        if isinstance(options[i], list):
                            option_str = 'Options: '+ ' '.join(['('+i for i in options[i]])    
                        if isinstance(options[i], str): 
                            option_str = options[i].replace(") ",")").replace("  ("," (").strip()
                    new_dict["option"] = option_str
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], option_str, None)
                elif args.dataset == 'CSQA': # 'CSQA', 'AQuA', 'date_understanding'   args.dataset == 'AQuA'
                    option_str = options[i].replace(") ",")").replace("  ("," (").strip()
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], option_str, None)
                    new_dict["option"] = option_str
                elif args.dataset == 'date_understanding': # 'CSQA', 'AQuA', 'date_understanding'   args.dataset == 'AQuA'
                    option_str = options[i].replace(") ",")").replace("  ("," (").strip()
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], option_str, None)
                    new_dict["option"] = option_str                     
                else: 
                    fin = get_Cot_generation_prompt(args.dataset, questions[i], None, None)

                new_dict["question"] = questions[i]
                
                if 'Llama' in args.api_name:
                    messages = [
                            {"role": "system", "content": "You are llama. You are a helpful assistant."},
                            {"role": "user", "content": fin}]
                    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                # 获取COT
                while True:
                    if 'gpt' in args.api_name:
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
                            new_dict["cot"] = str(res["Reasoning"])
                            new_dict["answer"]=res["Answer"]
                            break
                        except Exception as e:
                            print('gpt-3.5-turbo-1106','COT')
                            print(cleaned_res)
                            print(f"fails: {e}")
                            continue
                    if 'Llama' in args.api_name:
                        try:
                            generated_ids = model.generate(
                                    **model_inputs,
                                    max_new_tokens=1024,
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample=False
                                )
                            generated_ids = [
                                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                                        ]
                            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].replace("\n", " ")
                            if response.rstrip()[-1] != "}":
                                response = response + '}'
                            res = eval(response)

                            reasoning = str(res["Reasoning"]) 
                            if args.dataset == "AQuA":
                                
                                pattern = re.search(r"answer is\s*([A-E])\b", reasoning, re.IGNORECASE) # 匹配：the answer is A 或 B 或 C 或 D 或 E，可带前后空格和标点
                            else:
                                pattern = re.search(r"answer is\s*(\d+(?:\.\d+)?)", reasoning, re.IGNORECASE) # 匹配：the answer is 11.8 或 42 等数字（整数或小数）

                            if pattern:
                                answer_end = pattern.end()   # 截断到答案位置为止（含 "the answer is ..."）
                                return reasoning[:answer_end].strip()
                            else:
                                continue
                        except Exception as e:
                            print('Llama','COT')
                            print(response)
                            print(f"fails: {e}")
                            if 'Llama' in args.api_name:
                                break
                            continue
                if 'Llama' in args.api_name and 'cot' not in list(new_dict.keys()):
                    break
                new_dict["score"] = calc_ent(new_dict["cot"])  # 计算cot的信息熵
                f.write(json.dumps(new_dict) + '\n')
    
def calcu_self_cons_accuracy(args, times=5):
    '''
    对于生成的若干CoT，为每个example挑选最好的CoT
    '''
    # import pdb;pdb.set_trace()
    data_list = []
    root_path = os.path.join(args.output_dir + "/cot_generation/")
    files = os.listdir(root_path)
    for file in files:
        if args.type not in file:
            continue
        data_path = root_path + file
        data = []
        with open(data_path, 'r', encoding='utf-8') as r_f:
            for d in jsonlines.Reader(r_f):
                data.append(d)
            data_list.append(data)

    # data_list 长度为重复生成的次数（5）[[sample1, sample2, ...], [], [], [], []]
    cot_list=[]
    # 在所有生成的CoT中，为每一个prompt挑选最好的CoT
    for datalen in range(0, len(data_list[0])): 
        pred_candi = []
        for can_len in range(0,times):
            pred={}
            pred['question']=data_list[can_len][datalen]['question']
            if "option" in data_list[can_len][datalen]:
                pred['option'] =data_list[can_len][datalen]['option']
            pred['cot'] =data_list[can_len][datalen]['cot']
            pred['answer'] = data_list[can_len][datalen]['answer']
            
            if args.dataset in ['Addsub','MultiA', 'GSM8K']:
                match = re.search(r'\((\d+)\)', str(pred['answer']))  # match = re.search(r'\((\d+)\)', pred['answer'])
                if match:
                    pred['answer'] = match.group(1)
            
            pred['score'] = data_list[can_len][datalen]['score']
            pred_candi.append(pred)

        sorted_dicts = sorted(pred_candi, key=lambda x: x["score"], reverse=True)
        cot_pred=sorted_dicts[0]
        cot_list.append(cot_pred)

    # 对于不在cot record中的sample，需添加进record
    record_path = os.path.join(args.output_dir, f"cot_generation/{args.dataset}_{args.api_name}_cot_record_{args.pool_max}_threshold_{args.threshold}.jsonl")
    
    if os.path.exists(record_path):
        record = []
        with open(record_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                record.append(data)
        record_question = [i['question'] for i in record]
        with open(record_path, 'a', encoding='utf-8') as f:
            for i in cot_list:
                if i['question'] in record_question:
                    continue
                else:
                    f.write(json.dumps(i) + '\n')

    else:
        with open(record_path, 'w', encoding='utf-8') as f:
            for i in cot_list:
                f.write(json.dumps(i) + '\n')
    
    return cot_list

# 根据输入参数加载和处理不同的数据集
def get_data(args):
    dataset = args.dataset
    data_list = []
    question_list = []
    answer_list = []
    cot_list = []
    option_list = []
    if dataset in ['CSQA']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            # import pdb;pdb.set_trace()
            question = data['question']['stem']
            answer = data['answerKey']
            op = "Options: "
            for i in data['question']['choices']:
                op = op + " (" + i['label'] + ") " + i['text'] + " "
            option = op
            question_list.append(question)
            answer_list.append(answer)
            option_list.append(option)

    elif dataset in ['AQuA', 'MathQA']:
        with open(args.data_path, 'r', encoding='utf-8') as r_f:
            for data in jsonlines.Reader(r_f):
                data_list.append(data)
        for data in data_list:
            question = data['question']
            answer = data['correct']
            cot = data['rationale']
            opt = data['options']
            question_list.append(question)
            answer_list.append(answer)
            cot_list.append(cot)
            option_list.append(opt)

    elif dataset in ['GSM8K']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            question = data['question']
            question_list.append(question)
            answer = data['answer']
            match = re.search(r'#### (.*)', answer)
            if match:
                content_after_hash = match.group(1)
                answer_list.append(content_after_hash)
            else:
                answer_list.append(answer)
            last_newline_index = answer.rfind('\n')
            if last_newline_index != -1:
                answer= answer[:last_newline_index]
            cot_list.append(answer)

    elif dataset in ['MultiA', 'Addsub']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            question = data['sQuestion']
            answer = data['lSolutions'][0]
            cot = data['lEquations'][0]
            question_list.append(question)
            answer_list.append(answer)
            cot_list.append(cot)

    elif dataset in ['Strategyqa']:
        with open(args.data_path, "r", encoding='utf-8') as f:
            data_list = json.load(f)
        data_list = data_list['examples']
        for data in data_list:
            question = data['input']
            answer = "Yes" if data['target_scores']['Yes'] else "No"
            cot = data['target'].replace("Yes. ", "").replace("No. ", "")
            question_list.append(question)
            answer_list.append(answer)
            cot_list.append(cot)

    elif dataset in ['date_understanding']:
        with open(args.data_path, "r", encoding='utf-8') as f:
            data_list = json.load(f)['examples']
        for data in data_list:
            parts = data['input'].split("Options:\n")
            # The first part is the question
            question = parts[0].strip()
            option = "Options:  "+"  ".join(parts[1].split('\n'))
            match = re.search(r"\((\w)\)", data['target'])
            answer = match.group(1)
            question_list.append(question)
            option_list.append(option)
            answer_list.append(answer)

    return question_list, answer_list, cot_list, option_list

# Information Entropy 计算给定文本的信息熵
def calc_ent(word):
    """
        calculate shanno ent of x
    """
    data = word.split()  # 以空格分隔的字符串word
    x = np.array(data)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]  
        # 计算与x_value相同的单词数量除以x的总单词数量（即x.shape[0]），得到该单词在列表中的相对频率，即概率p
        logp = np.log2(p)
        ent -= p * logp
    return ent
  
def generate_prompt(args, idx, question, option, demo_pool,model,tokenizer):
    print(f'----------样本{idx}----------')
    demo_pool.demo_gen(args, question, option,model,tokenizer)
    questions, options = demo_pool.gather_scores(args, question)
    # import pdb; pdb.set_trace()
    # 现在CoT是全生成，对于train集中的数据只考虑question和option
    print("生成CoT...")
    
    # 删除了similarity
    cot_sc(args, questions, options, model,tokenizer,times=1) # 生成CoT
    final_list=calcu_self_cons_accuracy(args,times=1)

    prompt = ''
    if args.dataset in ['Strategyqa']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"
            prompt += "A: Let's think step by step. \n"
            prompt += i['cot'] + "\n"
            prompt += i['answer'] + '\n\n\n'
    
    if args.dataset in ['CSQA']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"
            prompt += str(i['option']) + "\n"
            prompt += "A: Let's think step by step. \n"
            prompt += str(i['cot']) + "\n"
            prompt += "Therefore, among A through E, the answer is ({})".format(i['answer']) + '\n\n\n'
    if args.dataset in ['date_understanding']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"
            prompt += str(i['option']) + "\n"
            prompt += "A: Let's think step by step. \n"
            prompt += str(i['cot']) + "\n"
            prompt += "Therefore, among A through F, the answer is ({})".format(i['answer']) + '\n\n\n'
                
    if args.dataset in ['MultiA', 'Addsub']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"                 
            prompt += "A: Let's think step by step. \n"
            prompt += str(i['cot']) + "\n"
            prompt += "Therefore, the answer is ({})".format(i['answer']) + '\n\n\n'
                        
    # labelled datasets   
    if args.dataset in ['GSM8K']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"
            prompt += "A: Let's think step by step. \n"
            prompt += i["cot"]+'\n'
            prompt += "Therefore, the answer is ({})".format(i["answer"]) +'\n\n\n'

    if args.dataset in ['AQuA']:
        for i in final_list:
            prompt += "Q: " + i['question'] + "\n"
            if 'Llama' in args.api_name:
                prompt += i['option'] + "\n"
            else:    
                prompt += "Options: " + i['option'] + "\n"
            prompt += "A: Let's think step by step. \n"
            prompt += i['cot'].replace('\n',' ') + "\n"
            prompt += "Therefore, among A through E, the answer is ({})".format(i['answer']) + '\n\n\n'
    return prompt
 
def main(args):
    # demo_pool = Pool(args)
    # demo_pool.initialize(args)
    
    data = Dataset(args.dataset, args.test_path).dataclass
    data_len = len(data.data)  # 数据长度
    infer = Inference(args.api_name, args.gpu)

    done_idxs=[]

    save_file = args.save_file
    file_dir = os.path.dirname(save_file)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if os.path.exists(save_file):
        f=open(save_file, "r",encoding="utf-8")
        lines=f.readlines()
        if data_len-len(lines)<data_len*0.05:
            os._exit(0)
        else:        
            for line in lines:
                line=line.strip()
                done_idxs.append(int(eval(line)["idx"]))
    logger.info(f"use model: {args.api_name}...")
    logger.info(f"save result in {save_file}...")
    start_idx=0
    model = None
    tokenizer = None
    if 'Llama' in args.api_name or 'Qwen' in args.api_name or 'deepseek' in args.api_name:
        if 'Llama' in args.api_name:
            logger.info(f"Load llama....")
        if 'Qwen' in args.api_name:
            logger.info(f"Load Qwen....")
        if 'deepseek' in args.api_name:
            logger.info(f"Load deepseek....")
        model = AutoModelForCausalLM.from_pretrained(
                                                    args.model_path,
                                                    torch_dtype="auto",
                                                    device_map="auto")
        logger.info(f"Load the tokenizer....")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if 'Mistral' in args.api_name:
        logger.info(f"Load Mistral....")
        model = Transformer.from_folder(args.model_path)

        logger.info(f"Load Mistral tokenizer....")
        tokenizer = MistralTokenizer.from_file(f"{args.model_path}/tokenizer.model.v3")

    
    for idx in tqdm(range(start_idx, data_len)):
        logger.info(idx)
        if idx in done_idxs:
            continue
        
        res_dict = {}
        
        if args.dataset in ["CSQA","date_understanding"]:
            question, answer, prompt, idx1 = data.get_question_by_idx(idx, args)
            # prompt, pred = infer.predict(question, option, args)
            res_dict["question"] = question
            res_dict["prompt"] = prompt
            res_dict["true_answer"] = answer
            res_dict["pred_answer"] = None
            res_dict["idx"] = idx
            # prompt = generate_prompt(args, idx, question, option, demo_pool,model,tokenizer)
            # prompt = res_dict["prompt"]
            f=open(save_file, "a",encoding="utf-8")
            infer.predict_with_api_service(question, None, args,res_dict,f,logger,prompt,model,tokenizer)
            
        elif args.dataset in ["GSM8K", "MultiA", "Strategyqa", "Addsub","MAWPS","ASDIV","SVAMP"]:
            question, truth_answer, prompt, idx1 = data.get_question_by_idx(idx,args)
            answer = ""
            res_dict["question"] = question
            res_dict["idx"] = idx
            res_dict["prompt"] = prompt
            res_dict["pred_answer"] = None
            res_dict["true_answer"] = truth_answer
            
            # prompt = generate_prompt(args, idx, question, '', demo_pool,model,tokenizer)
            # print(1)
            f=open(save_file, "a",encoding="utf-8")
            infer.predict_with_api_service(question, None, args, res_dict, f, logger, prompt, model,tokenizer)

        elif args.dataset == "AQuA":
            question, answer, prompt, idx = data.get_question_by_idx(idx, args)
            # option_str = ' '.join(['('+i for i in option])
            res_dict["question"] = question
            res_dict["prompt"] = prompt
            # res_dict["cot"] = cot
            res_dict["true_answer"] = answer
            res_dict["pred_answer"] = None
            res_dict["idx"] = idx
            # if 'Llama' in args.api_name:
            #     prompt = None
            # else:
            #     prompt = generate_prompt(args, idx, question, option, demo_pool,model,tokenizer)
            f=open(save_file, "a",encoding="utf-8")
            infer.predict_with_api_service(question, None, args, res_dict, f, logger, prompt,model,tokenizer)
    
if __name__ == '__main__':
    args = get_args()


    main(args)
    # dataset_score(args)