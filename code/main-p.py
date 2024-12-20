from dataload import Dataset
from inference import Inference
import torch
from tools import metric_max_over_ground_truths, f1_score, exact_match_score
import argparse
import itertools
import json
import jsonlines
import os,time
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from loguru import logger
import openai
from api_service import api_get_tokens
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.api_base = os.environ["OPENAI_API_BASE"]

def get_args():
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # openai.api_base = os.environ["OPENAI_API_BASE"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default="chatgpt")
    parser.add_argument("--train_path", type=str, default="data/CSQA/train_rand_split.jsonl")
    parser.add_argument("--test_path", type=str, default="data/CSQA/train_rand_split.jsonl")
    parser.add_argument("--data_path", type=str, default="data/CSQA/train_rand_split.jsonl")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument('--api_key_1106', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="CSQA",choices=["AQuA","GSM8K", "MultiA", "Addsub", "MathQA", "CSQA", "Strategyqa", "date_understanding","MAWPS","ASDIV","SVAMP"])
    parser.add_argument("--generate", default=True, action="store_true")
    parser.add_argument("--type", type=str)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--example_number", type=int, default=2)
    parser.add_argument("--error_check", default=False)
    parser.add_argument("--api_name", type=str,default=["gpt-3.5-turbo","gpt-4o-mini"])
    parser.add_argument("--save_file", type=str)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    return args


def accuracy(preds, golds):
    count = 0
    correct = 0
    for pred, gold in zip(preds, golds):
        prediction = pred["prediction"]
        gold = gold["answer"][0]
        if prediction == gold:
            correct += 1
        count += 1
    return {"accuracy": correct / count}


def gen_eval(preds, golds):
    em_total = 0
    f1_total = 0
    count = 0

    for pred, gold in zip(preds, golds):
        # Concatenate gold answers (can be multiple like NYT)
        sent = gold["question_sentence"].lower().strip()
        if "except" in sent[-10:]:
            continue
        count += 1
        golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
        golds = [" ".join(perm) for perm in list(itertools.permutations(golds))]
        prediction = pred["prediction"]
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, golds)
        f1_total += metric_max_over_ground_truths(f1_score, prediction, golds)

    return {"em": em_total / count, "f1": f1_total / count}

def run(args):
    
    data = Dataset(args.dataset, args.data_path).dataclass

    data_len = len(data.data)  # 数据长度
    infer = Inference(args.model, args.gpu)
    process_num=100
    process_num_count=100
    qps=30
    done_num=0
    done_idxs=[]
    for times in range(1):
        save_file = args.save_file
        print(save_file)
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
        logger.info(f"use model: {args.model}...")
        logger.info(f"save result in {save_file}...")
        start_idx=0
        llama_model = None
        tokenizer = None
        if 'Llama' in args.api_name or 'Qwen' in args.api_name or 'deepseek' in args.api_name:
            if 'Llama' in args.api_name:
                logger.info(f"Load llama....")
            if 'Qwen' in args.api_name:
                logger.info(f"Load Qwen....")
            if 'deepseek' in args.api_name:
                logger.info(f"Load deepseek....")
            llama_model = AutoModelForCausalLM.from_pretrained(
                                                        args.model_path,
                                                        torch_dtype="auto",
                                                        device_map="auto")
            logger.info(f"Load tokenizer....")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # if 'deepseek' in args.api_name:
        #     logger.info(f"Load deepseek....")
        #     llama_model = AutoModelForCausalLM.from_pretrained(
        #                                                 args.model_path,
        #                                                 torch_dtype=torch.bfloat16,
        #                                                 device_map="auto")
        #     logger.info(f"Load tokenizer....")
        #     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        #     llama_model.generation_config = GenerationConfig.from_pretrained(args.model_path)
        #     llama_model.generation_config.pad_token_id = llama_model.generation_config.eos_token_id
        
        if 'Mistral' in args.api_name:
            logger.info(f"Load Mistral....")
            llama_model = Transformer.from_folder(args.model_path)

            logger.info(f"Load Mistral tokenizer....")
            tokenizer = MistralTokenizer.from_file(f"{args.model_path}/tokenizer.model.v3")

        for idx in tqdm(range(start_idx, data_len)):
            start_time=time.time()
            logger.info(idx)
            if idx in done_idxs:
                # logger.info("skip "+str(idx))
                continue
            res_dict = {}
            
            if args.error_check:
                prompt, answer = data.get_question_by_idx(idx)
                _, pred = infer.predict(prompt, option=None, args=args)
                res_dict["question"] = prompt
                res_dict["idx"] = idx
                res_dict["pred_answer"] = pred
                res_dict["true_answer"] = answer
                with jsonlines.open(save_file, "a") as f:
                    f.write(res_dict)
            else:
                if args.dataset in ["CSQA","date_understanding"]:
                    question, answer, option = data.get_question_by_idx(idx)
                    # prompt, pred = infer.predict(question, option, args)
                    res_dict["question"] = question
                    res_dict["prompt"] = None
                    res_dict["true_answer"] = answer
                    res_dict["pred_answer"] = None
                    res_dict["idx"] = idx
                    f=open(save_file, "a",encoding="utf-8")
                    infer.predict_with_api_service(question, option, args,res_dict,f,logger,idx,llama_model,tokenizer)
                    

                elif args.dataset in ["GSM8K", "MultiA", "Strategyqa", "Addsub","MAWPS","ASDIV","SVAMP"]:
                    question, truth_answer = data.get_question_by_idx(idx)
                    answer = ""
                    # prompt, pred = infer.predict(sentence=question, option=None, args=args)
                    res_dict["question"] = question
                    res_dict["idx"] = idx
                    res_dict["prompt"] = None
                    res_dict["pred_answer"] = None
                    res_dict["true_answer"] = truth_answer
                    f=open(save_file, "a",encoding="utf-8")
                    infer.predict_with_api_service(question, None, args,res_dict,f,logger,idx,llama_model,tokenizer)

                elif args.dataset == "AQuA":
                    question, answer, cot, option = data.get_question_by_idx(idx)
                    res_dict["question"] = question
                    res_dict["prompt"] = None
                    res_dict["cot"] = cot
                    res_dict["true_answer"] = answer
                    res_dict["pred_answer"] = None
                    res_dict["idx"] = idx
                    f=open(save_file, "a",encoding="utf-8")
                    infer.predict_with_api_service(question, option, args,res_dict,f,logger,idx,llama_model,tokenizer)
            done_num+=1
            time.sleep(1/qps)
            process_num_count-=1
            
    while True:
        logger.info("wait for result.")
        f=open(save_file,"r")
        lines=f.readlines()
        if data_len-len(lines)<data_len*0.05:
            logger.info("ending......")
            time.sleep(40)
            os._exit(0)
            logger.info(f"finish {len(lines)} samples.")
            break
        # else:
        #     logger.info(count)
        f.close()

# 普通运行
def run_test(args):
    data = Dataset(args.dataset, args.data_path).dataclass
    infer = Inference(args.model, args.gpu)
    data_len = len(data.data)  # 数据长度
    logger.info(f"use model: {args.model}...")
    logger.info(f"save result in {args.save_file}...")

    lst = []
    count = 0
    correct  = 0
    preds = []
    golds = []

    print(args)
    file_dir = os.path.dirname(args.save_file)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(args.save_file):
        start_idx=0
    else:
        saved_data=[]
        with open(args.save_file,'r') as r_f:
            for data_i in jsonlines.Reader(r_f):
                saved_data.append(data_i)
        start_idx = saved_data[-1]["idx"]+1 if len(saved_data)!=0 else 0
        
    # import pdb;pdb.set_trace()
    for idx in tqdm(range(start_idx,data_len)):
        res_dict = {}
        # idx =data_len-1
        if idx==data_len-1:
            logger.info("over")
            os._exit(0)

        if args.dataset in ['GSM8K' , 'MultiA' , 'Strategyqa','Addsub']:
            question ,truth_answer= data.get_question_by_idx(idx)
            answer = ''
            prompt,pred = infer.predict(sentence=question,option=None,args=args)
            res_dict['idx'] = idx 
            res_dict['question'] = question
            res_dict['prompt'] = prompt
            res_dict['pred_answer'] = pred
            res_dict['true_answer'] = truth_answer
            with jsonlines.open(args.save_file, 'a') as f:
                f.write(res_dict)
            
            
        elif args.dataset=='AQuA':
            question, answer, cot , option =data.get_question_by_idx(idx)
            prompt,pred = infer.predict(question,option,args)
            res_dict['prompt'] = prompt
            res_dict['cot'] = cot
            res_dict['question'] = question
            res_dict['true_answer'] = answer
            res_dict['pred_answer'] = pred
            res_dict['idx'] = idx 
            with jsonlines.open(args.save_file, 'a') as f:
                f.write(res_dict)
            
        elif args.dataset in ['CSQA','date_understanding']:
            question, answer, option =data.get_question_by_idx(idx)
            prompt,pred = infer.predict(question,option,args)
            res_dict['prompt'] = prompt
            res_dict['question'] = question
            res_dict['true_answer'] = answer
            res_dict['pred_answer'] = pred
            res_dict['idx'] = idx 
            with jsonlines.open(args.save_file, 'a') as f:
                f.write(res_dict)

if __name__ == "__main__":
    
    # import pdb;pdb.set_trace()
    args = get_args()
    # run_est(args)
    run(args)
