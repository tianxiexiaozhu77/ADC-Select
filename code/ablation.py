# import re

# # 从文件中提取 idx 的函数
# def extract_idx_from_file(filepath):
#     idx_list = []
#     with open(filepath, 'r') as file:
#         for line in file:
#             match = re.match(r'idx:(\d+)', line)
#             if match:
#                 idx_list.append(int(match.group(1)))
#     return idx_list

# # 假设文件路径如下
# file_paths = [
#     '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation/GSM8K_all_gpt-3.5-turbo_1106_error_zero_shot_cot.log',
#     '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation/GSM8K_all_gpt-3.5-turbo_1106_error_hard_div.log',
#     '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation/GSM8K_all_gpt-3.5-turbo_1106_error_complex.log',
#     '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation/GSM8K_all_gpt-3.5-turbo_1106_error_IE_instance.log'
# ]

# # 提取所有文件的 idx 序号
# files_idx = [extract_idx_from_file(path) for path in file_paths]

# # 找出前三个均错误，最后一个正确的序号
# common_in_first_three = set(files_idx[0]).intersection(files_idx[1], files_idx[2])
# only_in_last = set(files_idx[3])
# result1 = common_in_first_three - only_in_last  # {675, 359, 75, 12, 494, 528, 187}

# # 找出前两个均错误，后两个均正确的序号
# common_in_first_two = set(files_idx[0]).intersection(files_idx[1])
# common_in_last_two = set(files_idx[2]).intersection(files_idx[3])
# result2 = common_in_first_two - common_in_last_two  # {98, 675, 359, 840, 75, 12, 462, 815, 528, 494, 754, 182, 184, 187, 798}

# # 找出第一个错误，后三个均正确的序号
# only_in_first = set(files_idx[0])
# common_in_last_three = set(files_idx[1]).intersection(files_idx[2], files_idx[3])
# result3 = only_in_first - common_in_last_three  # {768, 515, 1288, 267, 12, 1038, 782, 528, 1041, 530, 1042, 1046, 1048, 796, 797, 798, 1309, 1059, 806, 39, 40, 552, 809, 1067, 298, 46, 1070, 814, 815, 1075, 54, 1084, 576, 1088, 578, 835, 580, 837, 840, 75, 852, 597, 1113, 93, 353, 98, 611, 102, 359, 1129, 620, 368, 880, 380, 901, 649, 906, 652, 1165, ...}

# # 找出第一个和第三个错误，第二个和第四个均正确的序号
# common_in_first_and_third = set(files_idx[0]).intersection(files_idx[2])
# common_in_second_and_fourth = set(files_idx[1]).intersection(files_idx[3])
# result4 = (common_in_first_and_third - common_in_second_and_fourth) - result3 # {768, 515, 901, 1288, 649, 906, 267, 12, 652, 782, 1165, 528, 912, 1042, 403, 406, 1046, 1048, 1176, 796, 797, 1309, 672, 675, 1059, 298, 814, 1070, 1199, 1075, 439, 951, 187, 443, 1084, 1088, 962, 835, 580, 454, 75, 984, 1113, 353, 611, 102, 359, 489, 234, 1129, 236, 749, 494, 368, 510}

# # 找出第一个和第四个错误，第二个和第三个均正确的序号
# common_in_first_and_fourth = set(files_idx[0]).intersection(files_idx[3])
# common_in_second_and_third = set(files_idx[1]).intersection(files_idx[2])
# result5 = (common_in_first_and_fourth - common_in_second_and_third) - result3 - result4


# # 输出结果
# print("前三个均错误，最后一个正确的序号:", result1)
# print("前两个均错误，后两个均正确的序号:", result2)
# print("第一个错误，后三个均正确的序号:", result3)
# print("第一个和第三个错误，第二个和第四个均正确的序号:", result4)
# print("第一个和第四个错误，第二个和第三个均正确的序号:", result5)

import re

# 从文件中提取 idx 的函数
def extract_idx_from_file(filepath, adjust_by=0):
    idx_list = []
    with open(filepath, 'r') as file:
        for line in file:
            match = re.match(r'idx:(\d+)', line)
            if match:
                idx_list.append(int(match.group(1)) + adjust_by)
    return idx_list

file_paths = [
    '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation1/GSM8K_all_gpt-3.5-turbo_1106_error_zero_shot_cot.log',
    '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation1/GSM8K_all_gpt-3.5-turbo_1106_error_hard_div.log',
    '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation1/GSM8K_all_gpt-3.5-turbo_1106_error_complex.log',
    '/opt/data/private/zjx/ICL/inform/new_data/error_data/ablation1/GSM8K_all_gpt-3.5-turbo_1106_error_IE_instance.log'
]

        # {
        #     "name": "Python: test_gpt4o-mini",
        #     "type": "python",
        #     "request": "launch",
        #     "program": "/opt/data/private/zjx/ICL/inform/code/eval_em_f1.py",
        #     "console": "integratedTerminal",
        #     "justMyCode": true,
        #     "python": "/opt/conda/envs/p310/bin/python",
        #     "args": [
        #         "--dataset",
        #         "GSM8K",
        #         "--data_path",
        #         // "/opt/data/private/zjx/ICL/inform/new_data/ASDIV/zjx/inference/ASDIV_gpt-3.5-turbo-1106_COT_cotpromptgpt-4o-mini.jsonl",
        #         // "/opt/data/private/zjx/ICL/inform/new_data/GSM8K/zjx/inference_con_4omini/GSM8K_gpt-4o-mini_IE_instance.jsonl",
        #         // "/opt/data/private/zjx/ICL/inform/new_data/GSM8K/zjx/inference_con_4omini/GSM8K_gpt-4o-mini_hard_div_0.jsonl",
        #         "/opt/data/private/zjx/ICL/inform/new_data/GSM8K/zjx/inference_con_4omini/GSM8K_gpt-4o-mini_complex_0.jsonl",
        #         // "/opt/data/private/zjx/ICL/inform/new_data/ASDIV/zjx/inference/ASDIV_gpt-3.5-turbo-1106_hard_div_cotpromptgpt-4o-mini.jsonl",
        #         // "/opt/data/private/zjx/ICL/inform/new_data/ASDIV/zjx/inference/ASDIV_gpt-3.5-turbo-1106_complex_cotpromptgpt-4o-mini.jsonl",
        #         "--model",
        #         "gpt-3.5-turbo_1106",
        #     ],
        # },

# 提取所有文件的 idx 序号，其中 'file2.txt' 的 idx 全部加 1
files_idx = [
    extract_idx_from_file(file_paths[0]),
    extract_idx_from_file(file_paths[1], adjust_by=-1),
    extract_idx_from_file(file_paths[2]),
    extract_idx_from_file(file_paths[3])
]

# 找出前三个均错误，最后一个正确的序号
# all_list = files_idx[0] + files_idx[1] + files_idx[2]
common = list(set(files_idx[0]) & set(files_idx[1]) & set(files_idx[2]))
diff = list(set(common) - set(files_idx[3]))
print(diff)

common_in_first_three = set(files_idx[0]).intersection(files_idx[1], files_idx[2])
only_in_last = set(files_idx[3])
result1 = common_in_first_three - only_in_last  # {1088, 1059, 675, 580, 1075, 439, 510} 

# 找出前两个均错误，后两个均正确的序号
common_in_first_two = set(files_idx[0]).intersection(files_idx[1])
common_in_last_two = set(files_idx[2]).intersection(files_idx[3])
result2 = common_in_first_two - common_in_last_two # {1038, 530, 798, 1059, 675, 423, 428, 1075, 439, 1088, 580, 1227, 852, 620, 753, 754, 1016, 380, 510}

# 找出第一个错误，后三个均正确的序号
only_in_first = set(files_idx[0])
common_in_last_three = set(files_idx[1]).intersection(files_idx[2], files_idx[3])
result3 = only_in_first - common_in_last_three  # {7, 12, 652, 1038, 143, 528, 1041, 530, 153, 796, 798, 672, 162, 1059, 675, 37, 1190, 39, 40, 552, 806, 1067, 809, 423, 46, 1070, 815, 428, 1075, 54, 182, 184, 439, 187, 1084, 701, 62, 446, 576, 1088, 578, 580, 197, 837, 840, 75, 1227, 462, 464, 209, 852, 597, 87, 89, 218, 93, 353, 98, 610, ...}

# 找出第一个和第三个错误，第二个和第四个均正确的序号
common_in_first_and_third = set(files_idx[0]).intersection(files_idx[2])
common_in_second_and_fourth = set(files_idx[1]).intersection(files_idx[3])
result4 = (common_in_first_and_third - common_in_second_and_fourth) - result3

# 找出第一个和第四个错误，第二个和第三个均正确的序号
common_in_first_and_fourth = set(files_idx[0]).intersection(files_idx[3])
common_in_second_and_third = set(files_idx[1]).intersection(files_idx[2])
result5 = (common_in_first_and_fourth - common_in_second_and_third) - result3 - result4

# 输出结果
print("前三个均错误，最后一个正确的序号:", result1)
print("前两个均错误，后两个均正确的序号:", result2)
print("第一个错误，后三个均正确的序号:", result3)
print("第一个和第三个错误，第二个和第四个均正确的序号:", result4)
print("第一个和第四个错误，第二个和第三个均正确的序号:", result5)