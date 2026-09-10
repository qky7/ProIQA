#-----------------------------------------------------------------------

#该代码用于从XE数据种提取指定位置的知识点

#-----------------------------------------------------------------------

import json

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/share/LLM/Qwen3-Embedding-8B",
                        help="Path to the pretrained embedding model.")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Root directory containing dataset folders.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Dataset = "XES3G5M"
    
    with open(f'{args.data_dir}/{Dataset}/questions全集中文版.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)  # 假设这里读取的是顶层字典
        
    with open(f'{args.data_dir}/{Dataset}/XES3G5M_trees4_application.json', 'r', encoding='utf-8') as f:
        questions_trees = json.load(f)  # 假设这里读取的是顶层字典
        
    results = {}
    
    for id in questions_trees:
        results[id] = questions_trees[id]
        concepts = questions[id]['kc_routes'][0].split("----")
        results[id]['concept'] = concepts[2:4]
    
    with open(f'{args.data_dir}/{Dataset}/XES3G5M_trees4_application(include_concepts).json','w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,indent=4)
        