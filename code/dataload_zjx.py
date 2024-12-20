import json
import jsonlines
import os,re
import pickle
from tools import read_jsonl, answer2jsonl
import pandas as pd

'''
    dataload class 封装用于加载和处理特定dataset: 模块化/可重用
    - 数据集包括: GSM8K、AQuA、MultiA、Strategyqa、Date_understanding、CSQA、Addsub 
    - 提供访问单个问题及其相关答案，以及确定每个数据集中条目总数的功能
        - __init__(data_path): 基于给定路径，读入并初始化数据
        - get_question_by_idx(idx): 基于索引获取 question/answer/rationale/options/Solutions等具体数据内容(dataset specific)
        - get_data_length(): 获取数据集大小
'''

def restore_prompt(args, sorted_prompt):  # 恢复排序后的提示文本到原始顺序
    prompt_list_all = sorted_prompt.rstrip('\n\n').split('\n\n')
    prompt_list = prompt_list_all[:-1]
    test_q = prompt_list_all[-1]

    n = len(prompt_list)

    # 创建一个与 prompt_list 长度相同的空列表用于还原
    original_list = [None] * n
    left_index = 0
    right_index = n - 1

    for i in range(n):
        if i % 2 == 0:  # 偶数索引的值在排序时放到了右侧
            original_list[i] = prompt_list[right_index]
            right_index -= 1
        else:  # 奇数索引的值在排序时放到了左侧
            original_list[i] = prompt_list[left_index]
            left_index += 1

    # 将还原后的列表重新组合为字符串
    # original_prompt = "\n\n\".join(original_list) + "\n\n\"
    return original_list,test_q
    # return original_list[:args.example_number],test_q

def get_prompt_asc_order(args, prompt_list, test_q):  # 逆序
    new_list = prompt_list[::-1]

    prompt="\n\n".join(new_list)+"\n\n" + test_q +"\n"
    return prompt

def get_prompt_dec_order(args, prompt_list, test_q):  # 原来的顺序
    new_list = prompt_list

    prompt="\n\n".join(new_list)+"\n\n" + test_q +"\n"
    return prompt

def get_prompt_new(args, prompt_list, test_q):  # 对一个包含多个提示文本的字符串进行重新排序，并返回处理后的提示文本
    # prompt_list = prompt.rstrip('\n\n').split('\n\n')

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
    prompt="\n\n".join(new_list)+"\n\n" + test_q +"\n"
    return prompt

class GSM8K(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)
            # for data in jsonlines.Reader(r_f):
            #     self.data.append(data)

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q) 
        return question, answer, prompt, idx

        # if args.example_number != 10:
        #     prompt_ori, test_q = restore_prompt(args,prompt)
        #     prompt = get_prompt_new(args, prompt_ori, test_q)
        # return question, answer, prompt, idx
        # question = self.data[idx]['question']
        # answer = self.data[idx]['answer']
        # return question, answer

    def get_data_length(self):
        return len(self.data)


class AQuA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            # for data in jsonlines.Reader(r_f):
            #     self.data.append(data)
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx, args):
        # question = self.data[idx]['question']
        # answer = self.data[idx]['correct']
        # cot = self.data[idx]['rationale']
        # option = self.data[idx]['options']
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q) 
        return question, answer, prompt, idx
        # if args.example_number != 10:
        #     prompt_ori, test_q = restore_prompt(args,prompt)
        #     prompt = get_prompt_new(args, prompt_ori, test_q)
        # return question, answer, prompt, idx
        # return question, answer, cot, option

    def get_data_length(self):
        return len(self.data)


class MultiA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)
            # for data in jsonlines.Reader(r_f):
            #     self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q) 
        return question, answer, prompt, idx
        # if args.example_number != 10:
        #     prompt_ori, test_q = restore_prompt(args,prompt)  # 返回原始序列和问题
        #     prompt = get_prompt_new(args, prompt_ori, test_q)
        # return question, answer, prompt, idx
        # question = self.data[idx]['sQuestion']
        # answer = self.data[idx]['lSolutions']
        # return question, answer

    def get_data_length(self):
        return len(self.data)


class Strategyqa(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                dic = eval(line)
                self.data.append(dic)
            # self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        return question, answer, prompt, idx
        # question = self.data[idx]['input']
        # answer = 'Yes' if self.data[idx]['target_scores']['Yes'] else 'No'
        # return question, answer

    def get_data_length(self):
        return len(self.data)


class Date_understanding(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                dic = eval(line)
                self.data.append(dic)
            # self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        return question, answer, prompt, idx
        
        # parts = self.data[idx]['input'].split("Options:\n")
        # question = parts[0].strip()
        # option = "Options:  "+"  ".join(parts[1].split('\n'))
        # match = re.search(r"\((\w)\)", self.data[idx]['target'])
        # answer = match.group(1)
        # return question, answer, option

    def get_data_length(self):
        return len(self.data)


class CSQA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                dic = eval(line)
                self.data.append(dic)
            # for data in jsonlines.Reader(f):
            #     self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        return question, answer, prompt, idx
        # question = self.data[idx]['question']['stem']
        # answer = self.data[idx]['answerKey']
        # op = "Options: "
        # for i in self.data[idx]['question']['choices']:
        #     op = op + " (" + i['label'] + ") " + i['text'] + " "
        # return question, answer, op

    def get_data_length(self):
        return len(self.data)


class Addsub(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)

    def get_question_by_idx(self, idx, args):
        # question = self.data[idx]['sQuestion']
        # answer = self.data[idx]['lSolutions']
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q)
        # elif args.order == 'ours':

        return question, answer, prompt, idx

        # if args.example_number != 10:
        #     prompt_ori, test_q = restore_prompt(args,prompt)
        #     prompt = get_prompt_new(args, prompt_ori, test_q)


        # return question, answer, prompt, idx

    def get_data_length(self):
        return len(self.data)
class MAWPS(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)

    def get_question_by_idx(self, idx, args):
        # question = self.data[idx]['sQuestion']
        # answer = self.data[idx]['lSolutions']
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q)
        # elif args.order == 'ours':

        return question, answer, prompt, idx


    def get_data_length(self):
        return len(self.data)

class ASDIV(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)

    def get_question_by_idx(self, idx, args):
        # question = self.data[idx]['sQuestion']
        # answer = self.data[idx]['lSolutions']
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q)
        # elif args.order == 'ours':

        return question, answer, prompt, idx


    def get_data_length(self):
        return len(self.data)
    
class SVAMP(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for line in r_f:
                dic = eval(line)
                self.data.append(dic)

    def get_question_by_idx(self, idx, args):
        # question = self.data[idx]['sQuestion']
        # answer = self.data[idx]['lSolutions']
        question = self.data[idx]['question']
        answer = self.data[idx]['true_answer']
        idx = self.data[idx]['idx']
        prompt = self.data[idx]['prompt']
        if args.order == 'dec':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_dec_order(args, prompt_ori, test_q)
        elif args.order == 'asc':
            prompt_ori, test_q = restore_prompt(args,prompt)
            prompt = get_prompt_asc_order(args, prompt_ori, test_q)
        # elif args.order == 'ours':

        return question, answer, prompt, idx


    def get_data_length(self):
        return len(self.data)

# 用于处理错误数据(error_data)，数据格式{'prompt':prompt, 'true_answer':true_answer}
class ErrorData(object):
    def __init__(self, name):
        data_path = "data\\error_data\\" + name + "_" + "davinci003.jsonl"
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                self.data.append(data)

    def get_question_by_idx(self, idx, args):
        question = self.data[idx]['prompt']
        answer = self.data[idx]['true_answer']
        return question, answer

    def get_data_length(self):
        return len(self.data)

# 通用数据集类，根据提供的数据集名称和路径，动态创建相应数据集对象。
class Dataset(object):
    def __init__(self, data_name, data_path) -> None:
        self.data = data_path
        self.name = data_name
        self.dataclass = self.get_dataclass()

    # 使用get_dataclass方法根据数据集名称决定使用哪个具体的数据集类
    def get_dataclass(self):
        if self.data is None:
            return ErrorData(self.name)
        if self.name in ['GSM8K']:
            return GSM8K(self.data)
        elif self.name in ['AQuA']:
            return AQuA(self.data)
        elif self.name in ['MultiA']:
            return MultiA(self.data)
        elif self.name in ['Strategyqa']:
            return Strategyqa(self.data)
        elif self.name in ['date_understanding']:
            return Date_understanding(self.data)
        elif self.name in ['Addsub']:
            return Addsub(self.data)
        elif self.name in ['CSQA']:
            return CSQA(self.data)
        elif self.name in ['MAWPS']:
            return MAWPS(self.data)
        elif self.name in ['ASDIV']:
            return ASDIV(self.data)
        elif self.name in ['SVAMP']:
            return SVAMP(self.data)
        else:
            raise NotImplementedError
