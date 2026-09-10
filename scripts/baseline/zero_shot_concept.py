import os
# ============ GPU配置（修改此处设置使用的GPU） ============
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 如 "0"=单卡，"0,1"=双卡，"0,1,2,3"=四卡
# =========================================================

import json
from tqdm import tqdm
import re
import argparse
import ast
from sklearn.metrics import f1_score
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 本地模型配置（按需修改） ============
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "/inspire/hdd/project/ai4education/qianhong-p-qianhong/LLM/Qwen3-8B")  # 模型路径或HuggingFace模型名，如 "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512                # 最大生成token数
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


# 将知识点列表转换为二进制向量
def topics_to_binary_vector(topics_list, topic_dict, all_topics_list):
    vector = np.zeros(len(all_topics_list), dtype=int)
    for topic in topics_list:
        if topic in topic_dict:
            vector[topic_dict[topic]] = 1
        else:
            print(f"警告: 知识点 '{topic}' 不在全集中")
    return vector


if __name__ == "__main__":

    Dataset = "XES-500"
    model_name = LOCAL_MODEL_PATH.split("/")[-1]  # 自动从模型路径提取模型名
    full_concept_path = f"../../data/{Dataset}/All_concepts.json"
    Data_path = f"../../data/{Dataset}/concept_items.json"
    judge_path = f"../../output/LLMonTree_Concept_{model_name}_{Dataset}.json"

    prompt = """Please select the relevant concepts from the concept full set based on the following question text and its reasoning process (inference tree).
    [question text]:{}

    [reasoning process]:{}

    [concept full set]:{}

    Please output strictly according to the following format and do not include any additional information:
    ['XXX','XXX']
"""

    with open(Data_path, 'r', encoding='utf-8') as f:
        Data = json.load(f)

    with open(full_concept_path, 'r', encoding='utf-8') as f:
        full_concept = json.load(f)

    if os.path.exists(judge_path):
        with open(judge_path, 'r', encoding='utf-8') as f:
            judge = json.load(f)
        print("目标文件存在")
    else:
        judge = {}
        print("目标文件不存在")
    del judge["总体平均F1"]

    # 创建知识点到索引的映射
    topic_to_idx = {topic: i for i, topic in enumerate(full_concept)}

    batch_size = 32  # 批处理大小，根据GPU显存调整

    remain_item = list(Data.keys())
    remain_item = [id for id in remain_item if id not in judge.keys()]
    print("待处理的样本：", remain_item)

    num = 0
    sum_f1 = 0
    # 评估已有结果
    for id, tmp in judge.items():
        num += 1
        predict_label = topics_to_binary_vector(judge[id], topic_to_idx, full_concept)
        true_label = topics_to_binary_vector(Data[id]['concept'], topic_to_idx, full_concept)
        score = f1_score(true_label, predict_label, average='binary')
        # print(f"id:{id}，F1分数:", score)
        sum_f1 += score

    # 批处理推理
    for chunk_start in tqdm(range(0, len(remain_item), batch_size), desc='处理进度'):
        chunk_ids = remain_item[chunk_start:chunk_start + batch_size]

        # 构建当前批次的prompt
        chunk_contents = [
            prompt.format(Data[id]['question'], Data[id]['reasoning_tree'], list(full_concept.keys()))
            for id in chunk_ids
        ]

        # 一次 model.generate() 处理整批
        responses = batch_llm_inference(chunk_contents)

        for id, response in zip(chunk_ids, responses):
            if response is None:
                continue
            try:
                response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                output_concept = ast.literal_eval(response_clean)
                judge[id] = output_concept
                num += 1
                predict_label = topics_to_binary_vector(judge[id], topic_to_idx, full_concept)
                true_label = topics_to_binary_vector(Data[id]['concept'], topic_to_idx, full_concept)
                score = f1_score(true_label, predict_label, average='binary')
                print(f"id:{id}，F1分数:", score)
                sum_f1 += score
            except (IndexError, Exception) as e:
                pass

        # 每个batch结束后保存
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(judge, f, ensure_ascii=False, indent=2)
            print(f'已保存 {len(judge)} 个样本')

    print("\n\n----------------------------------------------------------------")
    print(f"处理的有效样本数量:{num}")
    print(f"平均F1: {sum_f1/num}")
    if '总体平均F1' in judge.keys():
        del judge['总体平均F1']
    sorted_judge = {id: v for id, v in sorted(judge.items(), key=lambda item: int(item[0]))}
    sorted_judge['总体平均F1'] = sum_f1/num
    with open(judge_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_judge, f, ensure_ascii=False, indent=2)
    print("结果文件已保存至：",judge_path)
