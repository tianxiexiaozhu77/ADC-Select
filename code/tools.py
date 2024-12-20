import datetime
import json
import jsonlines
import string
from collections import Counter


def read_jsonl(in_file):
    questions = []
    with open(in_file) as fin:
        for line in fin:
            question = json.loads(line)
            questions.append(question)
    return questions

# 确保每个问题至少有top_k个搜索结果：如果data_1没有足够结果，将回退到data_2来填补缺口。
def fall_back(data_1, data_2, top_k=5):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]
        datum_1["search_result"] = [article for article in datum_1["search_result"] if "text" in article]
        nb_retrieved = len(datum_1["search_result"])
        if nb_retrieved < top_k:
            datum_1["search_result"].extend(datum_2["search_result"][:top_k - nb_retrieved])
    return data_1

# 答案写入文件
def answer2jsonl(answers, questions, out_file, scores=None):
    # Confirm we have answers for all questions
    assert len(answers) == len(questions)
    if scores is not None:
        assert len(answers) == len(scores)
    outputs = []
    for q_idx in range(len(answers)):
        if scores is None:
            output = {"question_id": questions[q_idx]["question_id"], "prediction": answers[q_idx]}
        else:
            output = {"question_id": questions[q_idx]["question_id"], "prediction": answers[q_idx],
                      "score": scores[q_idx]}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)

# 检查两个数据列表是否长度相同，并且相应元素具有相同的question_id
def check_jsonl(data_1, data_2):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]

# 在句子的开头添加一个日期字符串，并以更易读的形式格式化日期
def add_today(sentence, date):
    date = datetime.datetime.strptime(date, '%Y/%m/%d')
    date = date.strftime("%B %d, %Y")
    sentence = "Today is {}. ".format(date) + sentence
    return sentence

# 通过转换为小写、移除标点、移除特定计数单位和修复空白字符来标准化答案，以便评估
def normalize_answer(s):
    def lower(text):  # 将文本转为小写
        return text.lower()

    def remove_punc(text):  # 移除所有标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # TODO: should we keep those counter removal?
    def remove_counter(text):  # 移除特定计数单位
        return text.replace("年", "").replace("月", "").replace("人", "").replace("日", "")

    def white_space_fix(text):  # 标准化文本中的空白字符
        return ' '.join(text.split())

    return white_space_fix(remove_counter(remove_punc(lower(s))))

# 计算精确匹配（EM）得分，用来判断标准化后的预测是否与标准化后的真实答案完全匹配。
def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# 计算标准化预测和真实答案之间的F1得分
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 计算给定度量（如EM或F1）对所有真实答案的最大得分
def metric_max_over_ground_truths(metric, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
