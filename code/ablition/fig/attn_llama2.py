from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm

data_path = '/opt/data/private/zjx/inform/data/Addsub_chatgpt_gpt-3.5-turbo-1106_IE_instance.jsonl'
prompt = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = eval(line)
        p = item['prompt']
        prompt.append(p)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/opt/data/private/zjx/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

plt.figure(figsize=(10, 6))


for i, prompt_text in enumerate(tqdm(prompt)):

    q_start_index = prompt_text.rfind("Question:")  # # 找到最后一段问题的起始位置
    if q_start_index == -1:
        continue  # 若找不到就跳过

    inputs = tokenizer(prompt_text, return_tensors='pt', return_offsets_mapping=True, padding=True)
    input_ids = inputs['input_ids'].to(device)  # # tokenization 时返回 offset mapping 以映射原文位置
    if input_ids.shape[-1] > 2000:  # 数据太长爆显存
        print(f'item {i} skipped.')
        continue
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs['offset_mapping'][0]  # shape: [seq_len, 2]
    
    q_id = None  # 找出token中首次出现q_start_index位置的token编号q_id
    for idx, (start, end) in enumerate(offset_mapping.tolist()):
        if start >= q_start_index:
            q_id = idx
            break
    if q_id is None or q_id == 0:
        continue

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    attn_layer = outputs.attentions[-1]  # list of [batch, heads, seq_len, seq_len] 最后一层注意力
    seq_len = input_ids.shape[1]
    last_token_index = seq_len - 1

    # shape: [heads, q_id]
    last_token_attn = attn_layer[0, :, last_token_index, :q_id]  # # 提取最后一个token对 [0:q_id] 所有token的注意力
    
    mean_attn_values = last_token_attn.mean(dim=0).cpu().numpy()  # # 对多个head求平均
    x_compressed = np.linspace(0, 100, num=len(mean_attn_values))
    
    plt.plot(x_compressed, mean_attn_values, linewidth=0.8, alpha=0.6, label=f"Prompt {i+1}")
    
    del outputs, attn_layer, last_token_attn, input_ids, attention_mask, inputs, x_compressed
    torch.cuda.empty_cache()
    gc.collect()

plt.title("Last Token Attention to Question Region", fontsize=14)
plt.xlabel("Token Position (Compressed 0–100)", fontsize=14)
plt.ylabel("Mean Attention Weight", fontsize=14)
plt.xticks([0, 50, 100], ["0", "50", "100"])
plt.ylim(0, 0.08)
plt.tight_layout()
plt.savefig("/opt/data/private/zjx/inform/fig/last_token_attention_to_qid1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# if __name__ == "__main__":


