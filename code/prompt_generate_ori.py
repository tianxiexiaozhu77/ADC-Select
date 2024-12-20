import torch
import jsonlines  
import json,re
from tqdm import tqdm
import numpy as np
import os,openai
import argparse,random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

# COT 生成prompt templete
def get_Cot_generation_prompt(dataset_name,question,option,answer):
    
    Cot_generation_prompt = {
    "CSQA" : str(
    "Q: " + question + "\n" + "Option:"+str(option) + "\n" +f"A: {str(answer)}\n"+ "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    ),
    "date_understanding" : str(
    "Q: " + question + "\n" + "Option:"+str(option) + "\n" + f"A: {str(answer)}\n" + "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer from provided options(A/B/C/D/E/F).\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"A\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"B\"}}\n"
    "Output: "
    ),
    "Addsub" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" + "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "MAWPS" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" + "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "ASDIV" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" + "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "SVAMP" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" + "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    ),
    "MultiA" : str(
    "Q: " + question + "\n" + f"A: {str(answer)}\n" +  "Let's think step by step."
    "Please output the detailed stepwise reasoning first and then output the answer in (arabic numerals) format.\n"
    "You must only output in a parsible JSON format. Two example outputs look like:\n"
    "Example 1: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (11.9).\"}}\n"
    "Example 2: {{\"Reasoning\": \"The detailed stepwise reasoning how to answer the question by think step by step.\", \"Answer\": \"Therefore, the answer is (20).\"}}\n"
    "Output: "
    )
    }
    return Cot_generation_prompt[dataset_name]

def get_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301')
    parser.add_argument("--dataset", type=str, default="CSQA",
                        choices=["AQuA", "GSM8K", "MultiA", "Addsub", "MathQA", "CSQA", "Strategyqa", "date_understanding","MAWPS","ASDIV","SVAMP"])
    parser.add_argument('--data_path', type=str, default='data/CSQA/train_rand_split.jsonl')
    parser.add_argument('--output_dir', type=str, default='data/CSQA/output')
    parser.add_argument('--type', type=str)
    parser.add_argument('--example_number', type=int, default=8)
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


def cot_sc(args,data_list):
    # import pdb;pdb.set_trace()
    # _, answer_list,_, _ = get_data(args)
    if args.dataset in ['CSQA','date_understanding','Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
        for cnt in range(5):
            target_path = os.path.join(args.output_dir,f"cot_generation/{args.dataset}_{args.type}_{cnt}.jsonl")
            loaded_question=[]
            if os.path.exists(target_path):
                with open(target_path, 'r', encoding='utf-8') as f:
                    for data in jsonlines.Reader(f):
                        loaded_question.append(data["question"])
            else :      
                file_dir = os.path.dirname(target_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                
            print(data_list[0])
            print(args)
            for i in tqdm(data_list):
                new_dict = {}
                # import pdb;pdb.set_trace()
                with open(target_path, "a", encoding='utf-8') as f:
                    if i["question"] in loaded_question:
                        continue
                    # if args.dataset in ['CSQA','date_understanding']:
                    #     fin = f"Q: {i['question']}\nOption: {str(i['option'])}\nLet's think step by step."
                    #     new_dict["option"] = i["option"]
                    # else: 
                    #     fin = f"Q: {i['question']}\nLet's think step by step."
                    if args.dataset in ['CSQA','date_understanding']:
                        if args.type == "complex":
                            fin = get_Cot_generation_prompt(args.dataset,i["question"],i["option"],None)
                        else :
                            fin = get_Cot_generation_prompt(args.dataset,i["question"],i["option"],i["answer"])
                        new_dict["option"] = i["option"]
                    else: 
                        if args.type == "complex":
                            fin = get_Cot_generation_prompt(args.dataset,i["question"],None,None)
                        else :
                            fin = get_Cot_generation_prompt(args.dataset,i["question"],None,i["answer"])
                    new_dict["question"] = i["question"]
                    new_dict["answer"]=i["answer"]
                    
                    # 获取COT
                    while True:
                        try:
                            # print(args.model)
                            response = openai.ChatCompletion.create(
                                    model=args.model,
                                    messages=[{'role': 'user', 'content': fin}],
                                    temperature=0.7,
                                    top_p=0.9,
                                    frequency_penalty=0.5,
                                    presence_penalty=0.5
                                )
                            cleaned_res = response['choices'][0]['message']['content'].replace('json', '').replace('```', '').replace('\n',' ')
                            
                            # new_dict["cot"]=cleaned_res
                            # new_dict["pred_answer"]=cleaned_res
                            res = eval(cleaned_res)
                            print(res)
                            new_dict["cot"] = str(res["Reasoning"])
                            new_dict["pred_answer"]=res["Answer"]
                            break
                        except Exception as e:
                            print(f"fails: {e}")
                            continue
                    if args.type == "complex":
                        new_dict["score"] = len(new_dict["cot"].split('\n'))
                    else :
                        new_dict["score"] = calc_ent(new_dict["cot"])
                    f.write(json.dumps(new_dict) + '\n')
            
            
def calcu_self_cons_accuracy(args):
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
    
    cot_list=[]
    for datalen in range(0, len(data_list[0])):
            pred_candi = []
            for can_len in range(0,5):
                pred={}
                pred['question']=data_list[can_len][datalen]['question']
                if "option" in data_list[can_len][datalen]:
                    pred['option'] =data_list[can_len][datalen]['option']
                pred['cot'] =data_list[can_len][datalen]['cot']
                pred['answer'] =data_list[can_len][datalen]['answer']
                pred['pred_answer'] =data_list[can_len][datalen]['pred_answer']
                if args.dataset in ['Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
                    match = re.search(r'\((\d+)\)', pred['pred_answer'])
                    if match:
                        pred['pred_answer'] = match.group(1)
                    
                pred['score'] = data_list[can_len][datalen]['score']
                pred_candi.append(pred)

            sorted_dicts = sorted(pred_candi, key=lambda x: x["score"], reverse=True)
            cnt=0
            index=0
            for idx,item in enumerate(sorted_dicts,0):
                
                if flexible_match(str(item['answer']),str(item['pred_answer'])):
                # if item['answer'] in item['pred_answer']:
                    if cnt==0:
                        index=idx
                    cnt+=1
            if cnt==0:
                cot_pred=sorted_dicts[0]
            else:
                cot_pred=sorted_dicts[index]
            cot_list.append(cot_pred)
            # print(cot_pred)

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

    elif dataset in ['MultiA', 'Addsub',"MAWPS","ASDIV","SVAMP"]:
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

    elif dataset in ['MAWPS']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            question = data['input']
            answer = data['target']
            # cot = data['lEquations'][0]
            question_list.append(question)
            answer_list.append(answer)
            cot_list.append(1)

    elif dataset in ['ASDIV']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            question = data['body'] + " " + data['question']
            answer = data['answer']
            cot = data['formula']
            question_list.append(question)
            answer_list.append(answer)
            cot_list.append(cot)

    elif dataset in ['SVAMP']:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)
        for data in data_list:
            question = data['Body'].rstrip(".") + ". " + data['Question']
            answer = data['Answer']
            cot = data['Equation']
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


# Question Selection----IE Ranking
# 根据不同的评分标准（如问题长度、信息熵等）为数据集中的问题打分，并将结果保存到文件中
def dataset_score(args):
    # import pdb;pdb.set_trace()
    print(f"dataset_score begin: type-{args.type}")
    question_list, answer_list, cot_list, option_list = get_data(args)
    score_list = []
    if args.type == 'complex' and args.dataset in ['CSQA','date_understanding','Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
            score_type = 'querylen'
    else :
            score_type = args.type
            
    for i in range(0, len(question_list)):
        if score_type == 'querylen':
            # import pdb;pdb.set_trace()
            score_list.append(len(question_list[i]))

        elif score_type == 'complex':
            cot = cot_list[i]
            # cot = cot.replace(" . ", "\n")
            # cot = cot.replace(". ", "\n")
            score_list.append(len(cot.split("\n")))

        else: 
            # score_type in ['easy', 'hard', 'hard_div', 'easy_div']:
            score_list.append(calc_ent(question_list[i]))
            
    idx_list = [i for i in range(0, len(question_list))]
    q_s = zip(idx_list, score_list)
    q_s = sorted(q_s, key=lambda x: x[1],reverse=True)

    print(f"sorted example:{q_s[0]}")
    sorted_path = os.path.join(args.output_dir, f"sorted/{args.type}_sorted.jsonl")
    if os.path.exists(sorted_path):
        return 
    sorted_file_dir = os.path.dirname(sorted_path)
    if not os.path.exists(sorted_file_dir):
        os.makedirs(sorted_file_dir)
    
    for idx in range(0, len(q_s)):
        i = q_s[idx][0]
        new_dict = {}
        new_dict["question"] = question_list[i]
        new_dict["answer"] = answer_list[i]
        if len(cot_list):
            new_dict["cot"] = cot_list[i]
        if len(option_list):
            new_dict['option'] = option_list[i]
        new_dict["score"] = q_s[idx][1]
        new_dict["idx"] = i
        with jsonlines.open(sorted_path,'a') as f:  
            f.write(new_dict)


# Question Selection----Diversity Pruning：计算两段文本之间的余弦相似度
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors)
    return similarity[0][1]


# Question Selection----Diversity Pruning+Order Augment：根据评分结果和数据集类型，生成提示文本并保存到文件中
def generate_prompt(args):
    print(f"generate_prompt begin:")
    sorted_path = os.path.join(args.output_dir, f"sorted/{args.type}_sorted.jsonl")
        
    datalist,mid_list,final_list = [],[],[]
    with open(sorted_path, 'r', encoding='utf-8') as r_f:
        for data in jsonlines.Reader(r_f):
            datalist.append(data)
            
    if args.type == 'ICL_random': 
        final_list = random.sample(datalist, args.example_number*4)
    elif args.type == 'ICL_IE'  : 
        final_list = datalist[:args.example_number*4]  
    elif args.type == 'ICL_IE_div'  : 
        mid_list = datalist
        idx = 0
        while len(final_list) <= args.example_number*2 and idx < len(mid_list):
            current_question = mid_list[idx]['question']
            is_similar = False
            for selected in final_list:
                if args.dataset in ["AQuA","Addsub","MAWPS","ASDIV","SVAMP"]:
                    similarity = 0.9
                elif args.dataset in ["date_understanding"]:
                    similarity = 0.7
                else :
                    similarity = 0.8
                if calculate_cosine_similarity(current_question, selected['question']) > similarity:
                    is_similar = True
                    break
            if not is_similar:
                final_list.append(mid_list[idx])
            idx += 1
        print(f"hard_div_candi_list size:{len(final_list)};\n example:{final_list[0]}")
    elif args.type == 'complex': 
        mm_list = datalist[:args.example_number]  
        if args.dataset in ['CSQA','date_understanding','Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
            cot_sc(args,mm_list)
            final_list=calcu_self_cons_accuracy(args)
        else :
            final_list = mm_list
    elif  args.type == 'hard_div': 
        mid_list = datalist
        mm_list = [] 
        idx = 0
        while len(mm_list) <= args.example_number*2 and idx < len(mid_list):
            current_question = mid_list[idx]['question']
            is_similar = False
            for selected in mm_list:
                if args.dataset in ["AQuA","Addsub","MAWPS","ASDIV","SVAMP"]:
                    similarity = 0.9
                elif args.dataset in ["date_understanding"]:
                    similarity = 0.7
                else :
                    similarity = 0.8
                if calculate_cosine_similarity(current_question, selected['question']) > similarity:
                    is_similar = True
                    break
            if not is_similar:
                mm_list.append(mid_list[idx])
            idx += 1
        print(f"hard_div_candi_list size:{len(mm_list)};\n example:{mm_list[0]}")
        
        mm_list = mm_list[:args.example_number*2]
        # import pdb;pdb.set_trace()
        if args.dataset in ['CSQA','date_understanding','Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
            cot_sc(args,mm_list)
            final_list=calcu_self_cons_accuracy(args)
        else :
            score_cot=[]
            for i in mm_list:
                score_cot.append(calc_ent(str(i['cot'])))
            idx_list = [i for i in range(0, len(mm_list))]
            # import pdb;pdb.set_trace()
            q_s = zip(idx_list, score_cot)
            q_s = sorted(q_s, key=lambda x: x[1],reverse=True)[:args.example_number]
            for idx in range(0, len(q_s)):
                i = q_s[idx][0]
                final_list.append(mm_list[i])
    else :
        final_list = datalist[:args.example_number*4]
        
    final_list = final_list[:args.example_number]
    print(f"{args.type}_final_list size:{len(final_list)};\n example:{final_list[0]}")
    
    if args.type == 'complex' and args.dataset in ['CSQA','date_understanding','Addsub','MultiA',"MAWPS","ASDIV","SVAMP"]:
        for i in tqdm(final_list):
            # if args.dataset in ['CSQA','date_understanding']:
            #     fin = f"Q: {i['question']}\nOption: {str(i['option'])}\nLet's think step by step."
            # else: 
            #     fin = f"Q: {i['question']}\nLet's think step by step."
            if args.dataset in ['CSQA','date_understanding']:
                fin = get_Cot_generation_prompt(args.dataset,i["question"],i["option"],None)
            else: 
                fin = get_Cot_generation_prompt(args.dataset,i["question"],None,None)
                # 获取COT
            while True:
                try:
                    response = openai.ChatCompletion.create(
                                    model=args.model,
                                    messages=[{'role': 'user', 'content': fin}],
                                    temperature=0.7,
                                )
                    cleaned_res = response['choices'][0]['message']['content'].replace('json', '').replace('```', '')
                    res = eval(cleaned_res)
                    i["cot"]  = str(res["Reasoning"])
                    break
                except Exception as e:
                    print(f"fails: {e}")
                    # time.sleep(1)
                    continue
        
    # import pdb;pdb.set_trace()
    target_path = os.path.join(args.output_dir,f"prompt/{args.dataset}_{args.type}_prompt.txt")
    selected_record_path = os.path.join(args.output_dir,f"prompt/{args.dataset}_{args.type}_selectedQ.txt")
    file_dir = os.path.dirname(target_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
            
    # unlabelled datasets: need to generate CoT
    if args.dataset in ['CSQA','date_understanding']:
        for i in final_list:
            with open(selected_record_path, "a", encoding='utf-8') as f:
                f.write(i['question'] + "\n\n\n")
            with open(target_path, "a", encoding='utf-8') as f:
                # print(i)
                f.write("Q: " + i['question'] + "\n")
                f.write(str(i['option']) + "\n")
                if 'ICL' not in args.type:
                    f.write("A: Let's think step by step. \n")
                    f.write(str(i['cot']) + "\n")
                f.write("Therefore, among A through F, the answer is ({})".format(i['answer']) + '\n\n\n') 
    if args.dataset in ['MultiA', 'Addsub',"MAWPS","ASDIV","SVAMP"]:
        for i in final_list:
            with open(selected_record_path, "a", encoding='utf-8') as f:
                f.write(i['question'] + "\n\n\n")
            with open(target_path, "a", encoding='utf-8') as f:
                f.write("Q: " + i['question'] + "\n")                    
                if 'ICL' not in args.type:
                    f.write("A: Let's think step by step. \n")
                    f.write(str(i['cot']) + "\n")
                f.write("Therefore, the answer is ({})".format(i['answer']) + '\n\n\n')
                        
    # labelled datasets   
    if args.dataset in ['GSM8K']:
        for i in final_list:
            with open(selected_record_path, "a", encoding='utf-8') as f:
                f.write(i['question'] + "\n\n\n")
            with open(target_path, "a", encoding='utf-8') as f:
                f.write("Q: " + i['question'] + "\n")
                if 'ICL' not in args.type:
                    f.write("A: Let's think step by step. \n")
                    f.write(i["cot"]+'\n')
                f.write("Therefore, the answer is ({})".format(i["answer"]) +'\n\n\n')

    if args.dataset in ['AQuA']:
            for i in final_list:
                with open(selected_record_path, "a", encoding='utf-8') as f:
                    f.write(i['question'] + "\n\n\n")
                with open(target_path, "a", encoding='utf-8') as f:
                    f.write("Q: " + i['question'] + "\n")
                    # print(i["option"])
                    result_dict = {item.split(')')[0]: item.split(')')[1] for item in i["option"]}
                    formatted_string = ' '.join(f"({key}) {value}" for key, value in result_dict.items())
                    f.write("Options: " + formatted_string + "\n")
                    if 'ICL' not in args.type:
                        f.write("A: Let's think step by step. \n")
                        f.write(i['cot'].replace('\n',' ') + "\n")
                    f.write("Therefore, among A through F, the answer is ({}), {}".format(i['answer'],result_dict[i['answer']]) + '\n\n\n')

    if args.dataset in ['Strategyqa']:
            for i in final_list:
                with open(selected_record_path, "a", encoding='utf-8') as f:
                    f.write(i['question'] + "\n\n\n")
                with open(target_path, "a", encoding='utf-8') as f:
                    f.write("Q: " + i['question'] + "\n")
                    if 'ICL' not in args.type:
                        f.write("A: Let's think step by step. \n")
                        f.write(i['cot'] + "\n")
                    f.write("Therefore, the answer (Yes or No) is ({})".format(i['answer']) + '\n\n\n')



if __name__ == '__main__':
    args = get_args()

    dataset_score(args)
    generate_prompt(args)
    
