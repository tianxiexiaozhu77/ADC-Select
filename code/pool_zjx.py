import os
import openai
import torch
import random
import json
import jsonlines
import re
from sentence_transformers import SentenceTransformer, models, util
from openai import OpenAI
import numpy as np
random.seed(42)

def is_single_dict_list(var):
    return isinstance(var, list) and len(var) == 1 and isinstance(var[0], dict)

def is_single_kv_dict(var):
    return isinstance(var, dict) and len(var) == 1


# 封装Pool的加载、配置和更新
class Pool(object):
    """
    只构建相似的question，如果是多选将包含选项，不包括答案和CoT
    流程：对于每一个question，首先计算pool中的相似个数，然后生成缺失的部分，填充进pool中
    """
    
    def __init__(self, args):
        self.pool_path = os.path.join(args.output_dir, "demo_pool")
        self.pool_file = f'{self.pool_path}/{args.api_name}_pool_{args.pool_max}_threshold_{args.threshold}.jsonl'
        self.question_list = []
        self.option_list = []
        self.IE_dict = {}
        self.client = OpenAI(base_url="https://www.gptapi.us/v1", api_key="********")
        
        # embedding
        self.model_path = '/opt/data/private/zyc/Models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
        self.word_embedding_model = models.Transformer(self.model_path)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.EMBEDDING = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model])

    def initialize(self, args):
        # TODO: 添加过量删除操作 根据IE值删除
        # TODO: pool的ablation 10000,7000,4000,1000
        print('----------初始化pool----------')
        if not os.path.exists(self.pool_path):
            os.makedirs(self.pool_path)
        
        indices = self._initialize_q(args)
        
        if args.dataset in ['AQuA', 'CSQA', 'date_understanding']:
            self._initialize_o(args, indices)
        
    def _initialize_q(self, args):
        '''
        初始化pool中的question
        '''
        
        indices = []
        if args.dataset in ["AQuA", "GSM8K", "CSQA"]:
            # 有train集，复制train集来初始化pool
            self.question_list, answer_list, cot_list, option_list = self.get_data(args)
            print(f"{args.dataset}有train集，大小为{len(self.question_list)}")
            
            if len(self.question_list)>(args.pool_max//2):
                indices = random.sample(range(len(self.question_list)), args.pool_max//2)
                self.question_list = [self.question_list[i] for i in indices]

        if not args.pool: # 不从指定的pool开始初始化
            print(f"pool初始化成功，有{len(self.question_list)}个demonstration (train集限制为{args.pool_max//2})")
        
        else: # 从指定的pool开始初始化
            with open(args.pool, 'r', encoding='utf-8') as f:
                for data in jsonlines.Reader(f):
                    self.question_list.append(data['question'])
            print(f"从指定pool初始化成功，有{len(self.question_list)}个demonstration (train集限制为{args.pool_max//2})")
            
        return indices
    
    def _initialize_o(self, args, indices):
        '''
        初始化pool中的options
        '''
        
        print(f'{args.dataset}为多选数据集，将保存选项信息')

        if args.dataset in ["AQuA", "CSQA"]:
            # 有train集，根据indices保存options
            _, _, _, self.option_list = self.get_data(args)
            if len(self.option_list)>5000:
                self.option_list = [self.option_list[i] for i in indices]

        if args.pool: # 从指定的pool开始初始化
            with open(args.pool, 'r', encoding='utf-8') as f:
                for data in jsonlines.Reader(f):
                    self.option_list.append(data['option'])
        
        print(f"选项初始化成功，有{len(self.option_list)}个option (train集限制为5000)")
        assert len(self.question_list) == len(self.option_list)
        
    def search_sim(self, args, question):
        """
        返回相似的question, option和对应的相似score
        """

        if not self.question_list:
            return None, None, None

        all_embeddings = self.EMBEDDING.encode(self.question_list, convert_to_tensor=True)
        input_embedding = self.EMBEDDING.encode([question], convert_to_tensor=True)
        cosine_similarities = util.pytorch_cos_sim(input_embedding, all_embeddings)

        top_k = len(self.question_list) if len(self.question_list) < args.example_number*4 else args.example_number*4
        values, indices = torch.topk(cosine_similarities[0], top_k, largest=True, sorted=True)
        values = values.tolist()
        indices = indices.tolist()

        if args.dataset in ['CSQA', 'AQuA', 'date_understanding']:
            return [self.question_list[i] for i in indices], [self.option_list[i] for i in indices], values
        else:
            return [self.question_list[i] for i in indices], [" "]*len(indices), values

    def demo_gen(self, args, question, option,model,tokenizer):
        
        top_q, _, values = self.search_sim(args, question)
        if values:
            mask = (np.array(values)>args.threshold) & (np.array(values)<1)
            count = mask.sum().item()
        else:
            count = 0
        
        needed = args.example_number-count
        print(f"在pool中找到{count}个相似sample，还需生成{needed if count<args.example_number else 0}个.")
        
        if needed <= 0:
            return 

        demo_generation_prompt = self.get_Demo_generation_prompt(args, question, option, needed)
        
        fails = 0
        if 'Llama' in args.api_name:
            messages = [
            {"role": "system", "content": "You are llama. You are a helpful assistant."},
            {"role": "user", "content": demo_generation_prompt}]
            text = tokenizer.apply_chat_template(
                                                messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
        while True:
            
            if 'gpt-3.5-turbo-1106' in args.api_name:
                try:
                    response = self.client.chat.completions.create(
                                model='gpt-4o-mini',
                                messages=[
                                    {
                                        "role": "user",
                                        "content": demo_generation_prompt
                                    }
                                ],
                                temperature=0.7,
                                top_p=0.9,
                                frequency_penalty=0.5,
                                presence_penalty=0.5
                            )

                    cleaned_res = response.choices[0].message.content.replace('```json\n', '').replace('\n```', '').replace('\n',' ')
                    # import pdb; pdb.set_trace()
                    if "}}" in cleaned_res and "{{" in cleaned_res:
                        res = eval(cleaned_res.replace("}}","}").replace("{{","{"))
                    else:
                        res = eval(cleaned_res)
                    # import pdb; pdb.set_trace()
                    # 防止不出现question关键词的错误
                    if 'demonstrations' in res:
                        question = res['demonstrations'][0]['question']

                    else:
                        if args.dataset in ['CSQA','date_understanding']:
                            if isinstance(res, dict) and len(res) == 2 and list(res.keys())==['question', 'Options']:
                                res['question']
                            else:
                                res[0]['question']
                        else:
                            if len(res)>1:
                                res[0]['question'] 
                            elif len(res) == 1:
                                
                                res['question']
                    self._add(args, res)
                    break
                except Exception as e:  # cleaned_res.replace("}}","}").replace("{{","{"))
                    print('gpt-3.5-turbo-1106','demo_gen')
                    print(cleaned_res)
                    print(f"fails: {e}")
                    fails += 1
                    if fails == 20:
                        print({"重复出现错误，退出."})
                        break
                    continue
            if 'Llama' in args.api_name:
                try:

                    generated_ids = model.generate(
                                    **model_inputs,
                                    max_new_tokens=2048,
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample=False,
                                )
                    generated_ids = [
                                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                                ]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if args.dataset == 'AQuA':
                        index = response.find('[')
                        response = response[index:]
                    res = eval(response)
                    if 'demonstrations' in res:
                        question = res['demonstrations'][0]['question']

                    else:
                        if args.dataset in ['CSQA','date_understanding','AQuA']:
                            if isinstance(res, dict) and len(res) == 2 and list(res.keys())==['question', 'Options']:
                                res['question']
                            else:
                                res[0]['question']
                        else:
                            if len(res)>1:
                                res[0]['question'] 
                            elif len(res) == 1 and is_single_kv_dict(res):
                                res['question']
                            elif len(res) == 1 and is_single_dict_list(res): 
                                res[0]['question']
                                res = res[0]                            
                    self._add(args, res)
                    break
                except Exception as e:  # cleaned_res.replace("}}","}").replace("{{","{"))
                    print('Llama','demo_gen')
                    print(response)
                    print(f"fails: {e}")
                    fails += 1
                    if fails == 200:
                        print({"重复出现错误，退出."})
                        break
                    continue
            
            
        # if needed:
        #     import pdb; pdb.set_trace()
        # self._add(args, res)
    
    def gather_scores(self, args, question):
        """
        根据近似出的top_q，计算相应IE score，缩放合并这两个score
        筛选出最终的question和option
        """
        self.IE_score()
        top_q, top_o, values = self.search_sim(args, question)
        
        ie_scores = []
        for i in top_q:
            ie_scores.append(self.IE_dict[i])
        
        # 放缩ie_score到nearest的值域
        near_min, near_max, ie_min, ie_max = min(values), max(values), min(ie_scores), max(ie_scores)
        if ie_max == ie_min:
            scaled_ie = ie_scores
        else:
            scaled_ie = [near_min + ((current-ie_min)/(ie_max-ie_min))*(near_max-near_min) for current in ie_scores]
        merged_score = []
        for i in range(len(top_q)):
            merged_score.append({'score': values[i]+scaled_ie[i], 'question': top_q[i], 'option': top_o[i],'sim_score':values[i],'ie_scores':ie_scores[i],'scaled_ie':scaled_ie[i]})
        
        merged_score.sort(key=lambda x: x['score'], reverse=True)
        questions =[i['question'] for i in merged_score][:args.example_number]
        options = [i['option'] for i in merged_score][:args.example_number]
        
        self._full_check(args)
        return questions, options

    def _full_check(self, args):
        if len(self.question_list) <= args.pool_max:
            return

        sorted_questions = sorted(enumerate(self.question_list), key=lambda x: self.IE_dict[x[1]], reverse=True)
        indices = [i for i, _ in sorted_questions[:args.pool_max]]
        self.question_list = [self.question_list[i] for i in indices]
        
        if args.dataset in ['CSQA', 'AQuA', 'date_understanding']:
            self.option_list = [self.option_list[i] for i in indices]  # 也对 option_list 进行相同的处理，确保 option_list 与 question_list 保持一致的元素数量
        # f=open(args.save_file, "w",encoding="utf-8")  # 不是这个文件
        # with open(self.pool_file, 'w', encoding='utf-8') as f: # self.question_list.append(str(res['demonstrations'][0]['question']))
        #     for i in 
        #     f.write(json.dumps({'question': res['demonstrations'][0]['question']}) + '\n')            

        print(f'pool超出给定最大大小{args.pool_max},删除超出部分，目前大小为{len(self.question_list)}')

    def _calc_ent(self, word):
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
        return float(ent)
    
    def IE_score(self):
        """
        根据不同的评分标准（如问题长度、信息熵等）为数据集中的问题打分
        """
        for i in range(0, len(self.question_list)):
            if self.question_list[i] not in self.IE_dict:
                self.IE_dict[self.question_list[i]] = self._calc_ent(self.question_list[i])

    def _add(self, args, res):
        # TODO: update至少要分多选和非多选，因为传参res不一样，可能需要为每一个数据集做对应有可能发生的情况
        if args.dataset in ['Strategyqa','Addsub','GSM8K', 'MultiA']:  # ['Strategyqa', 'Addsub']
            with open(self.pool_file, 'a', encoding='utf-8') as f:
                if 'demonstrations' in res:
                    self.question_list.append(str(res['demonstrations'][0]['question']))
                    f.write(json.dumps({'question': res['demonstrations'][0]['question']}) + '\n')            
                    self.question_list.append(str(res['demonstrations'][1]['question']))
                    f.write(json.dumps({'question': res['demonstrations'][1]['question']}) + '\n') 
                
                else: 
                    if len(res) == 1:
                        self.question_list.append(str(res["question"]))
                        f.write(json.dumps({'question': res["question"]}) + '\n')        
                    else:
                        for i in range(len(res)):
                            new_dict = {}
                            new_dict["question"] = str(res[i]["question"])
                            self.question_list.append(str(res[i]["question"]))
                            f.write(json.dumps(new_dict) + '\n')
            self._check()
        
        if args.dataset in ['AQuA','CSQA','date_understanding']:  
            with open(self.pool_file, 'a', encoding='utf-8') as f:
                if 'demonstrations' in res: # 2个
                    res = res['demonstrations']
                    for i in range(len(res)):
                        new_dict = {}
                        current_demo = res[i]["question"]
                        if args.dataset in ['CSQA','date_understanding']:
                            current_demo = res[i]["question"] +' Options: '+ res[i]["Options"]
                        question, option = self.demo_postprocess(args, current_demo)
                        
                        new_dict["question"] = question
                        new_dict["option"] = option
                        self.question_list.append(question)
                        self.option_list.append(option)
                        f.write(json.dumps(new_dict) + '\n')
                
                else:
                    if len(res) == 1 or args.dataset in ['CSQA','date_understanding','AQuA'] and isinstance(res, dict) and len(res) == 2 and list(res.keys())==['question', 'Options']: # 1个
                        new_dict = {}
                        current_demo = res["question"]
                        if args.dataset in ['CSQA','date_understanding','AQuA']:
                            current_demo = res["question"] +' Options: '+ res["Options"]
                        question, option = self.demo_postprocess(args, current_demo)
                        
                        new_dict["question"] = question
                        new_dict["option"] = option
                        self.question_list.append(question)
                        self.option_list.append(option)
                        f.write(json.dumps(new_dict) + '\n')
                        
                    else: # 3 or 4个
                        for i in range(len(res)):
                            new_dict = {}
                            current_demo = res[i]["question"]
                            if args.dataset in ['CSQA','date_understanding','AQuA']:  # ,'AQuA'
                                current_demo = res[i]["question"] +' Options: '+ res[i]["Options"]
                            # if args.dataset in ['AQuA']:  # ,'AQuA'
                            #     current_demo = res[i]["question"] + res[i]["Options"]
                            # if args.dataset in 'AQuA':
                            #     current_demo = res[i]["question"]
                                
                            question, option = self.demo_postprocess(args, current_demo)
                            
                            new_dict["question"] = question
                            new_dict["option"] = option
                            self.question_list.append(question)
                            self.option_list.append(option)
                            f.write(json.dumps(new_dict) + '\n')

            self._check()
            
    def _check(self):
        print(f"------> 更新：pool样例个数为{len(self.question_list)}")
    
    def demo_preprocess(self, args, question, option):
        if args.dataset == 'AQuA': # 'CSQA', 'AQuA', 'date_understanding'
            option_str = ''
            for op in option:
                option_str = option_str + op.replace(") ", ")") + " "
            # question = f'{question} Options: {option_str}'
            question = (question,option_str)
        if args.dataset == 'CSQA': # 'CSQA', 'AQuA', 'date_understanding'   args.dataset == 'AQuA'
            option_str = option.replace(") ",")").replace("  ("," (").strip()
            # for op in option:
            #     option_str = option_str + op.replace(") ", ")") + " "
            question = f'{question} {option_str}'
        if args.dataset == 'date_understanding': # 'CSQA', 'AQuA', 'date_understanding'   args.dataset == 'AQuA'
            option_str = option.replace(") ",")").replace("  ("," (").strip()
            # for op in option:
            #     option_str = option_str + op.replace(") ", ")") + " "
            question = f'{question} {option_str}'

        return question
    
    def demo_postprocess(self, args, demonstration):
        '''
        拆分多选生成demonstration时的question和option
        '''

        options = []
        if args.dataset in ['AQuA']:  # 'CSQA', 
            question, option_str = demonstration.split("Options: ")
            if 'Llama' in args.api_name:
                question, option_str = demonstration.split("Options: ")
                options = 'Options: ' + option_str 
            if 'gpt-3.5-turbo-1106' in args.api_name:
                pattern = r"([A-E])\)(.*?)(?=\s*[A-E]\)|\s*$)"
                matches = re.findall(pattern, option_str)
                options = [f"{match[0]}){match[1].strip()}" for match in matches]
        elif args.dataset == 'CSQA':
            question, option_str = demonstration.split("Options: ")
            options = 'Options: ' + option_str
        elif args.dataset == 'date_understanding':
            question, option_str = demonstration.split("Options: ")
            options = 'Options: ' + option_str
        # elif args.dataset == 'AQuA':
        #     question, option_str = demonstration.split("Options: ")
        #     options = 'Options: ' + option_str
            # pattern = r"(\([A-E]\))\s*([^()]*?)(?=\s*\([A-E]\)|$)"
            # matches = re.findall(pattern, option_str)
            # options = [f"{match[0]}{match[1].strip()}" for match in matches]
            
        else:
            question = demonstration

        return question, options
    
    def get_Demo_generation_prompt(self, args, question, option, count):  # 根据不同的数据集，设计输入GPT让它生成demo的提示
        # TODO: 多选prompt设计，question+option直接输入，然后通过后处理拆分出来分别update
        question = self.demo_preprocess(args, question, option)
        
        Demo_generation_prompt = {
        "CSQA" : str(
        f"Given question and options: \"{question}\"\n"    # 
        f"Please output {count} common sense question demonstrations related to this common sense question along with their corresponding options. "  # Please output {count} demonstrations with similar sentence structures and semantics. 
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: {{\"question\": \"Demonstration related to the given common sense question.\",\"Options\": \"Options A through E correspond to the demonstration, with one correct answer among them.\"}}\n"  # # "Example 1: {{\"question\": \"Demonstration including question and options you generated.\"}}\n"
        "Example 2: {{\"question\": \"Demonstration related to the given common sense question.\",\"Options\": \"Options A through E correspond to the demonstration, with one correct answer among them.\"}}\n"  # # "Example 2: {{\"question\": \"Demonstration including question and options you generated.\"}}\n"        
        "Output: "
        ),
        "date_understanding" : str(
        f"Given question and options: \"{question}\"\n" 
        f"Please output {count} common sense question demonstrations related to this common sense question along with their corresponding options. "  # f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"  # "You must only output in a parsible JSON format. Two example outputs look like: \n" 
        "Example 1: {{\"question\": \"Demonstration related to the given common sense question.\",\"Options\": \"Options A through F correspond to the demonstration, with one correct answer among them.\"}}\n"  # "Example 1: {{\"question\": \"Demonstration including question and options you generated.\"}}\n"
        "Example 2: {{\"question\": \"Demonstration related to the given common sense question.\",\"Options\": \"Options A through F correspond to the demonstration, with one correct answer among them.\"}}\n"  # "Example 2: {{\"question\": \"Demonstration including question and options you generated.\"}}\n"
        "Output: "
        ),
        "Addsub" : str(
        f"Given question: {question}\n" 
        f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: [{\"question\": \"Demonstration similar to the given question.\"}]\n"
        "Example 2: [{\"question\": \"Demonstration similar to the given question.\"}]\n"
        "Output: "
        ),
        "MultiA" : str(
        f"Given question: {question}\n" 
        f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Example 2: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Output: "
        ),
        "Strategyqa" : str(
        f"Given question: {question}\n" 
        f"Please output {count} common sense question demonstrations related to this common sense question. "  # f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Example 2: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Output: "
        ),
        "AQuA" : str(
        "Given question and options: " + "{\"question\": " + "\"" + f"{question[0]}"+"\"" +  " \"Options\": " + "\"" + f"{question[1]}" +"\"" +  "}\n"  # f"Given question and options: \"{question}\"\n" 
        f"Please output {count} math problems similar to this one, along with their corresponding options. "  # f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: [{\"question\": \"Demonstration similar to the given question.\",\"Options\": \"Options A through E correspond to the demonstration, with one correct answer among them.\"}]\n"  #  \"Demonstration including question and options you generated.
        "Example 2: [{\"question\": \"Demonstration similar to the given question.\",\"Options\": \"Options A through E correspond to the demonstration, with one correct answer among them.\"}]\n"
        "Output: "
        ),
        "GSM8K" : str(
        f"Given question: {question}\n" 
        f"Please output {count} demonstrations with similar sentence structures and semantics. "
        "You must only output in a parsible JSON format. Two example outputs look like: \n"
        "Example 1: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Example 2: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
        "Output: "
        ),
        }
        return Demo_generation_prompt[args.dataset]
    
    def get_data(self, args):
        dataset = args.dataset
        data_list = []
        question_list = []
        answer_list = []
        cot_list = []
        option_list = []
        if dataset in ['CSQA']:
            with open(args.train_path, 'r', encoding='utf-8') as f:
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
            with open(args.train_path, 'r', encoding='utf-8') as r_f:
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
            with open(args.train_path, 'r', encoding='utf-8') as f:
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
            with open(args.train_path, 'r', encoding='utf-8') as f:
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
            with open(args.train_path, "r", encoding='utf-8') as f:
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
            with open(args.train_path, "r", encoding='utf-8') as f:
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