import json
import jsonlines
import random

def transform(input_file,output_file):
    # 读取 JSON 数据
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 写入 JSONL 数据
    with jsonlines.open(output_file, mode='w') as writer:
        for item in data:
            writer.write(item)
   
   
def split_dataset(root_dir,file_path):
    # 读取 JSONL 文件
    data = []
    with jsonlines.open(root_dir+file_path) as reader:
        for obj in reader:
            data.append(obj)

    # 打乱数据
    random.shuffle(data)

    # 计算切分点
    split_index = int(0.7 * len(data))

    # 切分数据
    train_data = data[:split_index]
    test_data = data[split_index:]

    # 保存训练集
    with jsonlines.open(root_dir+"train.jsonl", mode='w') as writer:
        for item in train_data:
            writer.write(item)

    # 保存测试集
    with jsonlines.open(root_dir+"test.jsonl", mode='w') as writer:
        for item in test_data:
            writer.write(item)

    print("数据已分割并保存。")
         
# 定义输入和输出文件路径
root_dir = 'inform/new_data/MultiA/'
json_file_path = 'MultiArith.json'
jsonl_file_path = 'MultiArith.jsonl'

# transform(root_dir+json_file_path,root_dir+jsonl_file_path)

split_dataset(root_dir,jsonl_file_path)