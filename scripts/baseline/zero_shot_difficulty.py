import os
# ============ GPU配置（修改此处设置使用的GPU） ============
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 如 "0"=单卡，"0,1"=双卡，"0,1,2,3"=四卡
# =========================================================

import json
from tqdm import tqdm
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 本地模型配置（按需修改） ============
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "/inspire/hdd/project/ai4education/qianhong-p-qianhong/LLM/Qwen3-8B")  # 模型路径或HuggingFace模型名，如 "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 64                 # 最大生成token数（难度评估输出很短）
# ===============================================

temperature = 1.0
top_p = 0.7
max_retry = 10

# ============ 加载模型（device_map="auto" 自动适配多GPU） ============
_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_gpu_ids = [x.strip() for x in _cuda_visible.split(",") if x.strip()]
print(f"CUDA_VISIBLE_DEVICES={_cuda_visible}，使用 {len(_gpu_ids)} 张GPU")

print(f"正在加载模型: {LOCAL_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("模型加载完成")

def batch_llm_inference(contents):
    """批量推理：一次 model.generate() 处理多条prompt"""
    messages_list = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": c}]
        for c in contents
    ]
    texts = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pad_token_id
        )

    input_len = inputs.input_ids.shape[1]
    responses = [tokenizer.decode(output[input_len:], skip_special_tokens=True) for output in outputs]
    return responses
# ================================================================


def extract_number_regex(text, tag="<tag>"):
    """使用正则表达式提取数字"""
    # 方法2.1: 使用原始字符串和tag
    pattern = re.escape(tag) + r'(\d+)' + re.escape(tag)
    match = re.search(pattern, text)

    # 方法2.2: 更通用的正则表达式（匹配任何标签内的数字）
    # pattern = r'<(\w+)>(\d+)<\1>'
    # match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从 '{text}' 中提取数字")


if __name__ == "__main__":

    Dataset = "Algebra"
    model_name = LOCAL_MODEL_PATH.split("/")[-1]  # 自动从模型路径提取模型名

    origin_path = f"../../data/{Dataset}/items.json"
    Data_path = f"../../data/{Dataset}/difficulty_items.json"
    judge_path = f"../../output/LLMonTree_Difficulty_{model_name}_{Dataset}.json"

    prompt = """Please evaluate the difficulty level of this question within the range [1,5] based on the following question text and its reasoning process (reasoning tree). The value can fall on either end of the range and must be an integer. A higher value indicates a higher difficulty level.
    [question text]:{}

    [reasoning process]:{}

    Please output strictly according to the following format and do not include any additional information(The content enclosed within the <tag> is the numerical value you assess):
    <tag>X<tag>

"""

    with open(Data_path, 'r', encoding='utf-8') as f:
        Data = json.load(f)

    with open(origin_path, 'r', encoding='utf-8') as f:
        origin_Data = json.load(f)

    if os.path.exists(judge_path):
        with open(judge_path, 'r', encoding='utf-8') as f:
            judge = json.load(f)
        print("目标文件存在")
    else:
        judge = {}
        print("目标文件不存在")

    batch_size = 8  # 批处理大小，根据GPU显存调整

    remain_item = list(Data.keys())
    remain_item = [id for id in remain_item if id not in judge.keys()]
    print("待处理的样本：", remain_item)

    num = 0
    sum_acc = 0
    sum_wacc = 0
    # 评估已有结果
    for id, tmp in judge.items():
        num += 1
        predict_label = int(judge[id])
        true_label = int(re.findall(r'\d+', origin_Data[id]['level'])[0])
        print(f"id:{id}，真实标签：{true_label}，预测标签：{predict_label}")
        if predict_label == true_label:
            sum_acc += 1
            sum_wacc += 1
        elif abs(predict_label - true_label) == 1:
            sum_wacc += 0.6

    # 批处理推理
    for chunk_start in tqdm(range(0, len(remain_item), batch_size), desc='处理进度'):
        chunk_ids = remain_item[chunk_start:chunk_start + batch_size]

        # 构建当前批次的prompt
        chunk_contents = [
            prompt.format(Data[id]['question'], Data[id]['reasoning_tree'])
            for id in chunk_ids
        ]

        # 一次 model.generate() 处理整批
        responses = batch_llm_inference(chunk_contents)

        for id, response in zip(chunk_ids, responses):
            if response is None:
                continue
            try:
                response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                predict_label = extract_number_regex(response_clean)
                print("提取出的内容:", predict_label)
                judge[id] = predict_label
                num += 1
                true_label = int(re.findall(r'\d+', origin_Data[id]['level'])[0])
                print(f"id:{id}，真实标签：{true_label}，预测标签：{predict_label}")
                if predict_label == true_label:
                    sum_acc += 1
                    sum_wacc += 1
                elif abs(predict_label - true_label) == 1:
                    sum_wacc += 0.6
            except (IndexError, Exception) as e:
                pass

        # 每个batch结束后保存
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(judge, f, ensure_ascii=False, indent=2)
            print(f'已保存 {len(judge)} 个样本')


    print(f"处理的有效样本数量:{num}")
    print(f"平均ACC: {sum_acc/num}")
    print(f"平均WACC: {sum_wacc/num}")
    if '总体平均ACC' in judge.keys():
        del judge['总体平均ACC']
    if '总体平均WACC' in judge.keys():
        del judge['总体平均WACC']
    sorted_judge = {id: v for id, v in sorted(judge.items(), key=lambda item: int(item[0]))}
    sorted_judge['总体平均ACC'] = sum_acc/num
    sorted_judge['总体平均WACC'] = sum_wacc/num
    with open(judge_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_judge, f, ensure_ascii=False, indent=2)
    print("结果文件已保存至：",judge_path)
