#--------------------------------------------------------------
#本代码用于配套DeAR方法中预处理数据时，配置每个测试样本的分解样例
#--------------------------------------------------------------
import json 
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity

#处理如何读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行JSON对象
            obj = json.loads(line.strip())
            data.append(obj)
    return data

#处理从作答过程中提取子问题
def extract_questions(input_text):
    # 首先按换行符拆分文本为多个小句
    sentences = input_text.split('\n')
    
    # 用于存储提取出的问题语句
    sublist = []
    
    # 遍历每个小句，提取以问号结尾的问题语句
    for sentence in sentences:
        # 查找句子中第一个问号的位置
        question_mark_index = sentence.find('?')
        
        # 如果找到问号，则提取从开头到问号的部分作为问题
        if question_mark_index != -1:
            question = sentence[:question_mark_index + 1].strip()
            sublist.append(question)
    
    return sublist

def extract_embedding(Model, tokenizer, texts, batch_size=50, max_length=300):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = Model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu()
        
        embeddings.append(embedding)
        
    return torch.cat(embeddings)
    
device = "cuda:3" if torch.cuda.is_available() else "cpu"

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
    #model_path = "/inspire/hdd/project/ai4education/tongjunkai-p-tongjunkai/LLM/BERT/bert-large-uncased"
    model_path = args.model_path
    Dataset = 'XES3G5M_know'
    
    train_Data = read_jsonl('{args.data_dir}/GSM8K/train_socratic.jsonl')
    if Dataset == 'XES3G5M':
        with open(f'{args.data_dir}/{Dataset}/questions_translated.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
        with open(f'{args.data_dir}/{Dataset}/question_picture_text.json', 'r', encoding='utf-8') as f:
            test_picture_text = json.load(f)
    elif Dataset == 'MATH':
        with open(f'{args.data_dir}/{Dataset}/items.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
    elif Dataset == 'XES3G5M_application':
        with open(f'{args.data_dir}/XES3G5M/application_questions_translated.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
    elif Dataset == 'XES3G5M_550':
        with open(f'{args.data_dir}/XES3G5M/questions_550.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
    elif Dataset == 'TAL':
        with open(f'{args.data_dir}/TAL/TAL_sample1000.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
    elif Dataset == 'XES3G5M_2000':
        with open(f'{args.data_dir}/XES3G5M/XES3G5M_questions_translated_2000.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)
    elif Dataset =='XES3G5M_know':
        with open(f'{args.data_dir}/XES3G5M/xe_know_translated.json', 'r', encoding='utf-8') as f:
            test_Data = json.load(f)

    else:
        print("数据集无法导入!!!")
        exit()
        
    
    #首先将每道训练样本提取其中的所有子问题，预处理成子问题列表
    processed_train = {}
    for index, sample in enumerate(train_Data):
        subquestions = extract_questions(sample['answer'])
        processed_train[index] = {
            'content':sample['question'],
            'subquestions':subquestions
        }
    print(processed_train[0])
    
    #导入BERT模型
    BERT_tokenizer = BertTokenizer.from_pretrained(model_path)
    BERT_Model = BertModel.from_pretrained(model_path, device_map="auto",)
    
    #聚合所有训练题目文本
    train_question_text = []
    for index, sample in processed_train.items():
        train_question_text.append(sample['content'])
        
    train_question_embedding = extract_embedding(BERT_Model, BERT_tokenizer, train_question_text)
    
    #聚合所有测试有样本题目文本
    test_question_text = []
    if Dataset == 'TAL':
        for sample in test_Data:
            test_question_text.append(sample['problem'].replace('$',''))
    else:
        for index, sample in test_Data.items():
            if Dataset == 'XES3G5M':
                if index in test_picture_text.keys():
                    cleaned_text = sample['content_en'].split('question_')[0]
                    test_question_text.append(test_picture_text[index]+ " " + cleaned_text)
                else:
                    test_question_text.append(sample['content_en'])
            elif Dataset == 'MATH':
                test_question_text.append(sample['problem'])
            elif Dataset == 'XES3G5M_application':
                test_question_text.append(sample['content'])
            elif Dataset == 'XES3G5M_550' or Dataset == 'XES3G5M_2000':
                test_question_text.append(sample['translated_content'])
            elif Dataset == 'XES3G5M_know':
                test_question_text.append(sample['question'])
            else:
                print("处理测试数据有误!!!")
                exit()
        
    test_question_embedding = extract_embedding(BERT_Model, BERT_tokenizer, test_question_text)
    
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(test_question_embedding, train_question_embedding)
    
    # 为每个测试样本选择三个相似度分数最高的训练样本
    top_3_indices = {}
    for index, row in enumerate(similarity_matrix):
        top_3 = list(row.argsort()[-3:][::-1])
        top_3 = [int(i) for i in top_3]
        top_3_indices[index] = top_3
    
    #print("top_3_indices:",top_3_indices)
    
    #构建最终测试样本分解样例
    test_split_examples = {}
    if Dataset == 'TAL':
        for index, sample in enumerate(test_Data):
            test_split_examples[sample['qid']] = []
            for i, prob in enumerate(top_3_indices[index]):
                test_split_examples[sample['qid']].append({f"Example question {i}:":processed_train[prob]['content'],"Decomposition:":processed_train[prob]['subquestions']})
            
    else:
        for index, id in enumerate(test_Data.keys()):
            test_split_examples[id] = []
            for i, prob in enumerate(top_3_indices[index]):
                test_split_examples[id].append({f"Example question {i}:":processed_train[prob]['content'],"Decomposition:":processed_train[prob]['subquestions']})
            
    
    # 定义要保存的JSON文件路径
    if Dataset == 'XES3G5M_application':
        save_file_path = f'{args.data_dir}/XES3G5M/application_test_split_examples.json'
    elif Dataset == 'XES3G5M_550':
        save_file_path = f'{args.data_dir}/XES3G5M/questions_550_split_examples.json'
    elif Dataset == 'TAL':
        save_file_path = '{args.data_dir}/TAL/TAL_sample1000_split_examples.json'
    elif Dataset == 'XES3G5M_2000':
        save_file_path = f'{args.data_dir}/XES3G5M/questions_2000_split_examples.json'
    elif Dataset == 'XES3G5M_know':
        save_file_path = '{args.data_dir}/XES3G5M/xe_know_translated_split_examples.json'
    else:
        save_file_path = f'{args.data_dir}/{Dataset}/test_split_examples.json'

    # 打开文件并将字典写入JSON文件
    with open(save_file_path, 'w', encoding='utf-8') as f:
        # indent参数用于美化输出，使JSON文件更易读
        json.dump(test_split_examples, f, ensure_ascii=False, indent=4)

    print(f"字典已保存到 {save_file_path}")
    
    
        
    
    

    
        
    
    

    
    
        