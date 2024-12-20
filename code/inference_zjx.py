import os
import openai
import torch
import random
from transformers import GenerationConfig  
from api_service import api_get_tokens
from prompt_generate import get_Cot_generation_prompt
os.environ["TOKENIZERS_PARALLELISM"] = "true"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ["OPENAI_API_BASE"]
import json

OPENAI_KEYS_list = [

]
api_count=random.randint(0,100)



def split_by_last_occurrence(text, delimiter):
    # 使用 rpartition 将字符串按最后一次出现的 delimiter 分割
    before, sep, after = text.rpartition(delimiter)
    # 如果找到了 delimiter，则返回分割后的两个部分
    if sep:
        return before, sep + after
    # 如果没有找到 delimiter，则返回原始字符串和空字符串
    return text, ""

# 从文件系统加载特定数据集的提示文本
def get_prompt(args,llama_model,tokenizer,option):
    # import pdb;pdb.set_trace()
    if 'gpt' in args.api_name:
        output_dir = os.path.dirname(os.path.dirname(args.save_file))
        prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_prompt.txt")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data=data.split('\n\n\n')
        data=data[:args.example_number]
        data="\n\n".join(data)+"\n\n"
        
    if 'Llama' in args.api_name:        
        output_dir = os.path.dirname(os.path.dirname(args.save_file))
        prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_llama_prompt.txt")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data=data.split('\n\n\n')
        cleaned_data = [item for item in data if item]
        data=cleaned_data[:args.example_number]
        data="\n\n".join(data)+"\n\n"
        
        
        # # # 生成文本
        # output_dir = os.path.dirname(os.path.dirname(args.save_file))
        # prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_selectedQ.txt")
        # with open(prompt_path, 'r', encoding='utf-8') as f:
        #     data = f.read()
        # data=data.split('\n\n\n')
        # data=data[:args.example_number]
        # prompt_path_save = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_llama_prompt.txt")
        # with open(prompt_path_save, 'a', encoding='utf-8') as f_save:
        #     if args.dataset in ["Addsub","AQuA"]:
        #         responses = []
        #         for i,d in enumerate(data):
        #             if i == 0:
        #                 continue
        #             fin = get_Cot_generation_prompt(args.dataset, d, option, None)
        #             messages = [
        #                         {"role": "system", "content": "You are llama. You are a helpful assistant."},
        #                         {"role": "user", "content": fin}]
        #             text = tokenizer.apply_chat_template(
        #                                         messages,
        #                                         tokenize=False,
        #                                         add_generation_prompt=True
        #                                     )
        #             model_inputs = tokenizer([text], return_tensors="pt").to(llama_model.device)
        #             generated_ids = llama_model.generate(
        #                             **model_inputs,
        #                             max_new_tokens=1024,
        #                             pad_token_id=tokenizer.eos_token_id,
        #                             do_sample=False,
        #                         )
        #             generated_ids = [
        #                                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        #                             ]
        #             response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
        #             responses.append(response.replace('\n',''))
                    
        #             f_save.write('\n\n\nQ: '+ d + '\n')
        #             f_save.write('A: Let\'s think step by step. \n')
        #             res = eval(response.replace('\n',''))               
        #             if "Therefore, the answer is " in res["Reasoning"]:
        #                 res["Reasoning"], res["Answer"] = split_by_last_occurrence(res["Reasoning"], "Therefore")                                            
        #             f_save.write(res["Reasoning"])
        #             if "Answer" in list(res.keys()):  #  '\n' + 
        #                 answer = res["Answer"].strip(' ()')
        #                 if answer.replace('.', '', 1).isdigit() and answer.count('.') <= 1:
        #                     if not res["Answer"].strip().endswith('.'):
        #                         res["Answer"] = res["Answer"].strip() + '.'  #  + '.\n\n\n'
        #                     f_save.write('\n' + "Therefore, the answer is " + res["Answer"])
        #                 else:
        #                     if not res["Answer"].strip().endswith('.'):
        #                         res["Answer"] = res["Answer"].strip() + '.'
        #                     f_save.write('\n' + res["Answer"])  #  + '\n\n\n'
        #             f_save.flush()
        #             # print('Q: '+ d + '\n')
        #             # print('A: Let\'s think step by step. \n')
        #             # print(res["Reasoning"] + '\n')
        #             # if "Answer" in list(res.keys()):
        #             #     answer = res["Answer"].strip(' ()')
        #             #     if answer.replace('.', '', 1).isdigit() and answer.count('.') <= 1:
        #             #         print("Therefore, the answer is " + res["Answer"])
        #             #     else:
        #             #         print(res["Answer"] + '\n\n\n')
        # print(1)
                       
    return data

# 封装模型的加载、配置和生成逻辑
class Inference(object):
    def __init__(self, model=None, device=0, generate=False, example=None):
        self.model = model 
        self.bot = None  
        self.hug_classifier = None  
        self.device = device  
        self.examples = example 

    # 根据不同模型与数据集动态构建AI模型输入提示，并添加问题及引导信息
    def generate_input(self, sentence, option, args,llama_model,tokenizer): 
        # if 'gpt' in args.api_name:
        fin_input = "" 
        if args.dataset in ['GSM8K', 'MultiA', 'Strategyqa', 'date_understanding', 'Addsub']:
            if args.type not in ["COT"]:
                inentropy_prompt = get_prompt(args,llama_model,tokenizer,None)
                fin_input = inentropy_prompt
                
            if args.dataset == 'Strategyqa':
                fin_input = fin_input + "Question: " + sentence + "\nAnswer: Let's think step by step.\n" 
                if args.type in ["COT"]:
                    fin_input = fin_input + "\nPlease give the final answer in Yes or No, as 'Therefore, the answer is (Yes or No).'" 
            elif args.dataset == 'date_understanding':
                fin_input = fin_input + "Question: " + sentence + "\n"+ option + "\nAnswer: Let's think step by step.\n" 
                if args.type in ["COT"]:
                    fin_input = fin_input + "\nPlease give the correct answer from the provided options, as 'Therefore, among A through F, the answer is (A/B/C/D/E/F).'" 
            else:
                fin_input = fin_input + "Question: " + sentence + "\nAnswer: Let's think step by step.\n" 
                if args.type in ["COT"]:
                    fin_input = fin_input + "\nPlease give the final answer in arabic numerals, as 'Therefore, the answer is (5.0).'" 
                
        elif args.dataset in ['AQuA']:
            if args.type not in ["COT"]:
                inentropy_prompt = get_prompt(args,llama_model,tokenizer,option)
                fin_input = inentropy_prompt
            
            question = sentence
            option_str = "\nOptions: "
            for op in option:
                option_str = option_str + "(" + op.replace(" )", ")") + " "
            fin_input = fin_input + "Question: " + question + "\n"+ option_str + "\nAnswer: Let's think step by step.\n"  
            if args.type in ["COT"]:
                    fin_input = fin_input + "\nPlease give the correct answer from the provided options, as 'Therefore, among A through F, the answer is (A/B/C/D/E/F).'" 
        elif args.dataset in ['CSQA']:
            if args.type not in ["COT"]:
                inentropy_prompt = get_prompt(args)
                fin_input = inentropy_prompt
            fin_input = fin_input + "\n Question: " + sentence + "\n" + option + "\nAnswer: Let's think step by step.\n" 
            if args.type in ["COT"]:
                    fin_input = fin_input + "\nPlease give the correct answer from the provided options, as 'Therefore, among A through F, the answer is (A/B/C/D/E).'" 
        return fin_input

    def predict_gpt(self, input, taskname=None):  
        print(input)
        output = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                        {"role": "user", "content": input }
                    ],
            temperature=0
        )
        
        pred = output['choices'][0]['message']['content']
        
        return pred

    # 基于已加载的模型对输入的文本进行预测
    def predict(self, sentence, option, args):  
        ind = 0
        if args.error_check:  
            while True:
                try:
                    input = ""
                    openai.api_key = OPENAI_KEYS_list[random.randint(0,100)% len(OPENAI_KEYS_list)]
                    pred = self.predict_gpt(sentence)
                    break
                except Exception as e:
                    print(f"fails: {e}")
                    openai.api_key = OPENAI_KEYS_list[random.randint(0,100)% len(OPENAI_KEYS_list)]
                    input = ""
                    pred = self.predict_gpt(sentence)
                    if ind > 10:
                        return -1
                    ind += 1
                    continue
        else:
                global api_count
                input = self.generate_input(sentence, option, args)
                while True:
                    try:
                        pred = self.predict_gpt(input)
                        break
                    except:
                        api_count = (api_count + 1) % len(OPENAI_KEYS_list)
                        openai.api_key = OPENAI_KEYS_list[api_count]
                        pred = self.predict_gpt(input)
                        # time.sleep(3)
                        if ind > 10:
                            return input,"None"
                        ind += 1
                        continue
        return input, pred
    
    def predict_with_api_service(self, sentence, option, args,res_dict,f,logger,idx,model,tokenizer):
        ind = 0
        global api_count
        input = self.generate_input(sentence, option, args,model,tokenizer)
        res_dict["prompt"]=input
        api_get_tokens(args.api_name,input,res_dict,f,logger,args,model,tokenizer)



