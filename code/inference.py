import os
import openai
import torch
import random
from transformers import GenerationConfig  
from api_service_zjx import api_get_tokens
from prompt_generate import get_Cot_generation_prompt
os.environ["TOKENIZERS_PARALLELISM"] = "true"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ["OPENAI_API_BASE"]

OPENAI_KEYS_list = [

]
api_count=random.randint(0,100)


# 从文件系统加载特定数据集的提示文本
def get_prompt(args):
    # import pdb;pdb.set_trace()
    output_dir = os.path.dirname(os.path.dirname(args.save_file))
    prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_{args.prompt}_prompt.txt")  # cot是gpt-4-turbo生成的
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = f.read()
    data=data.split('\n\n\n')
    data=data[:args.example_number]
    data="\n\n".join(data)+"\n\n"
    # if 'gpt' in args.api_name:
    #     output_dir = os.path.dirname(os.path.dirname(args.save_file))
    #     prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_prompt_1129.txt")
        
    #     with open(prompt_path, 'r', encoding='utf-8') as f:
    #         data = f.read()
    #     data=data.split('\n\n\n')
    #     data=data[:args.example_number]
    #     data="\n\n".join(data)+"\n\n"
    # if 'Llama' in args.api_name:        
    #     output_dir = os.path.dirname(os.path.dirname(args.save_file))
    #     prompt_path = os.path.join(output_dir,f"prompt/{args.dataset}_{args.type}_llama_prompt.txt")
        
    #     with open(prompt_path, 'r', encoding='utf-8') as f:
    #         data = f.read()
    #     data=data.split('\n\n\n')
    #     cleaned_data = [item for item in data if item]
    #     data=cleaned_data[:args.example_number]
    #     data="\n\n".join(data)+"\n\n"
    
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
    def generate_input(self, sentence, option, args): 
        fin_input = "" 
        if args.dataset in ['GSM8K', 'MultiA', 'Strategyqa', 'date_understanding', 'Addsub',"MAWPS","ASDIV","SVAMP"]:
            if args.type not in ["COT"]:
                inentropy_prompt = get_prompt(args)
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
                inentropy_prompt = get_prompt(args)
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
        input = self.generate_input(sentence, option, args)  # ,model,tokenizer
        res_dict["prompt"]=input
        api_get_tokens(args.api_name,input,res_dict,f,logger,args,model,tokenizer)



