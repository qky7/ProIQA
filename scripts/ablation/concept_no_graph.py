import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.metrics import f1_score
import random
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from sklearn.model_selection import KFold  # 新增KFold导入
from collections import defaultdict
import gc

#--------------------------------------------------------------------------

# XE数据集的知识点预测任务，使用五折交叉方式，并优化验证流程提高计算效率，在classifier_train1_copy8_2的基础上，将构建表征放在五折循环外部
# 修改版本：将推理树文本整体传入Qwen3-8B提取隐藏状态，替代原有的DGL图神经网络表征构建方式
#--------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)

def format_reasoning_tree_text(tree_data):
    """将推理树结构直接序列化为带节点序号的文本，用于输入Qwen3-8B"""
    lines = [
        "The following is a reasoning tree for a math problem. "
        "Each node represents a question or sub-question with its reasoning step."
    ]
    for node_id in sorted(tree_data.keys(), key=int):
        node = tree_data[node_id]
        parent = node.get('parent', 'None')
        content = node.get('content', '')
        rationale = node.get('rationale', '')
        lines.append(
            f"Node {node_id} (parent: {parent}): "
            f"[Question] {content} [Reasoning] {rationale}"
        )
    return "\n".join(lines)

def last_token_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx]
        }

# 特征提取函数
def extract_features(model, tokenizer, texts, batch_size=20, max_length=400,use_tqdm=False):
    model.eval()
    features = []

    iterator = range(0, len(texts), batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc="提取表征进度:")

    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            #计算所有有效token的平均
            last_features = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            batch_features = last_features.cpu().float()

        features.append(batch_features)
        del inputs, outputs, last_features  # 显式删除变量
        torch.cuda.empty_cache()  # 清理缓存
        torch.cuda.synchronize()  # 等待 GPU 操作完成，确保缓存清理生效

    return torch.cat(features)

def generate_dataset1(id_key, all_question_trees, full_set_concepts, problem_feat_dict, concept_feat_dict, tree_dict):
    features = []
    trees = []
    labels = []
    ids = []

    for id in tqdm(id_key, desc=f"处理进度:"):
        n = 0
        tmp_tree = tree_dict[id]
        positive_concepts = all_question_trees[id]['concept']
        negative_concepts = [c for c in full_set_concepts if c not in positive_concepts]

        # for c in negative_concepts:
        for c in random.sample(negative_concepts, len(negative_concepts)//2):
            # 拼接问题特征和概念特征（直接从字典获取）
            feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
            features.append(feat)
            trees.append(tmp_tree)
            labels.append(0)
            ids.append(id)
            n += 1

        while n > 0:
            for c in positive_concepts:
                if n == 0:
                    break
                feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
                features.append(feat)
                trees.append(tmp_tree)
                labels.append(1)
                ids.append(id)
                n -= 1

    features = torch.stack(features)
    labels = torch.tensor(labels)

    # 创建数据集和数据加载器
    processed_dataset = ReasoningTreeDataset(trees=trees,text_features=features, labels=labels,ids=ids)

    return processed_dataset



def generate_dataset2(id_key, all_question_trees, full_set_concepts, problem_feat_dict, concept_feat_dict, tree_dict):
    features = []
    trees = []
    labels = []
    ids = []
    for id in tqdm(id_key, desc=f"构建验证集"):
        tmp_tree = tree_dict[id]
        positive_concepts = all_question_trees[id]['concept']

        # 遍历所有概念（与原逻辑一致）
        for c in full_set_concepts:
            feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
            features.append(feat)
            trees.append(tmp_tree)
            labels.append(1 if c in positive_concepts else 0)
            ids.append(id)
    # 转换为张量
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return ReasoningTreeDataset(trees=trees, text_features=features, labels=labels, ids=ids)


def collate_fn1(batch):
    """
    自定义数据拼接函数，处理推理树文本表征（不再使用DGL图）
    """
    tree_features = torch.stack([item["tree"] for item in batch]).to(device)
    text_features = torch.stack([item["text_feature"] for item in batch]).to(device)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float).to(device)
    ids = [item['id'] for item in batch]

    return {
        "tree_features": tree_features,  # 推理树文本的Qwen3编码表征
        "text_features": text_features,
        "labels": labels,
        "ids": ids
    }

class ReasoningTreeDataset(Dataset):
    def __init__(self, trees, text_features, labels, ids):
        self.trees = trees  # 列表，每个元素是推理树文本的Qwen3编码表征 (dim,) 的张量
        self.text_features= text_features
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        return {
            "tree": self.trees[idx],  # 推理树文本表征张量
            "text_feature": self.text_features[idx],  # 题目文本的原始表征
            "label": self.labels[idx],             # 难度值标签
            "id":self.ids[idx]
        }


class SaMerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_hidden_layers = 3,dropout_rate=0.2, activation=nn.SiLU):
        super(SaMerClassifier, self).__init__()

        self.layer_sizes = [input_dim,1024,512,256,128]

        layers = []

        for i in range(len(self.layer_sizes)-1):
            pre_dim = self.layer_sizes[i]
            curr_dim = self.layer_sizes[i+1]

            layers.append(nn.Linear(pre_dim, curr_dim))
            # 添加批归一化层
            layers.append(nn.BatchNorm1d(curr_dim))
            # 添加激活函数
            layers.append(activation())
            # 添加dropout层
            layers.append(nn.Dropout(dropout_rate))

        # 添加输出层
        layers.append(nn.Linear(self.layer_sizes[-1], output_dim))
        layers.append(nn.Sigmoid())

        # 将所有层组合成序列
        self.scoring_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.scoring_layer(x)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/share/LLM/Qwen3-Embedding-8B",
                        help="Path to the pretrained embedding model.")
    parser.add_argument("--data_dir", type=str, default="../../data",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--dataset", type=str, default="XES-1600",
                        choices=["XES-500", "XES-1600"], help="Dataset name.")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    batch_size = 4000    # 
    max_length = 500
    tree_max_length = 1024  # 推理树文本最大长度
    num_epochs = 70
    learning_rate = 2e-5
    seed = 46
    #分类模型参数
    classify_num_hidden_layers = 0
    classify_activation=nn.SiLU
    classify_dropout_rate = 0.4
    classify_output_dim = 1
    n_splits = 5  # 5折交叉验证
    n_sample = 20

    generate_model = True
    # 加载模型和Tokenizer (device_map='auto' 自动分布到4张GPU)
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
    base_model = AutoModel.from_pretrained(model_path,
                                            device_map='auto',
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True,
                                            # attn_implementation="flash_attention_2"
                                            )

    print("模型导入成功")

    # 冻结主模型参数
    for param in base_model.parameters():
        param.requires_grad = False

    Dataset = args.dataset
    # Dataset_filepath = f'{args.data_dir}/{Dataset}/new_{Dataset}_trees3_application(include_concepts).json'
    Dataset_filepath = f'{args.data_dir}/{Dataset}/concept_items.json'


    with open(Dataset_filepath, 'r', encoding='utf-8') as f:
        all_question_trees = json.load(f)  # 假设这里读取的是顶层字典

    to_remove = []
    full_set_concepts = set()
    for id, sample in all_question_trees.items():
        if sample['concept'] == []:
            to_remove.append(id)
            print(f"id: {id}知识点不存在")
        if len(sample['reasoning_tree']) <= 1:
            to_remove.append(id)
            print(f"id: {id}推理树节点数量<=1")
            continue
        full_set_concepts.update(sample['concept'])

    full_set_concepts = sorted(list(full_set_concepts))
    print("知识点全集数量:",len(full_set_concepts))

    # 步骤2: 提取并处理键（可选择是否随机打乱）
    unique_key = set(all_question_trees.keys())
    unique_key = list(unique_key)
    print("清洗前样本总数:",len(unique_key))
    print("因知识点为空要删除的样本id:",to_remove)
    for id in to_remove:
        unique_key.remove(id)

    unique_key.sort()
    print("清洗后样本总数:",len(unique_key))

    # 分割训练集和测试集
    # 打乱数据顺序
    # 设置随机种子
    set_seed(seed)
    random.shuffle(unique_key)

    template1 = """
    Instruct: Provide a textual description of a question, and retrieve the concepts contained in this question.\n
    Query:
    question text:{}
    """

    all_epoch_results = defaultdict(list)

    problem_texts_all = []
    problem_ids_all = []
    for id in unique_key:
        cont = template1.format(all_question_trees[id]['question'])
        problem_texts_all.append(cont)
        problem_ids_all.append(id)
    # 提取所有问题文本特征并映射到ID
    problem_feats = extract_features(base_model, tokenizer, problem_texts_all, use_tqdm=True)
    problem_feat_dict = {id: feat for id, feat in zip(problem_ids_all, problem_feats)}

    # 预计算所有概念的特征
    print("预计算所有概念特征...")
    concept_feats = extract_features(base_model, tokenizer, full_set_concepts, use_tqdm=True)
    concept_feat_dict = {c: feat for c, feat in zip(full_set_concepts, concept_feats)}

    # 预计算所有推理树的特征（改为：格式化树文本 → Qwen3编码）
    print("预计算所有推理树特征（文本编码方式）...")
    tree_texts = []
    tree_ids_for_encoding = []
    for id in tqdm(unique_key, desc="格式化推理树文本"):
        tree_text = format_reasoning_tree_text(all_question_trees[id]['reasoning_tree'])
        tree_texts.append(tree_text)
        tree_ids_for_encoding.append(id)

    tree_feats = extract_features(base_model, tokenizer, tree_texts,
                                  max_length=tree_max_length, use_tqdm=True)
    tree_dict = {id: feat for id, feat in zip(tree_ids_for_encoding, tree_feats)}


    dim = base_model.config.hidden_size

    # 1. 删除模型引用（解除对模型的内存占用）
    del base_model
    # 如果tokenizer也不再需要，可一并删除
    del tokenizer

    # 2. 强制Python垃圾回收（清理已删除对象的残留引用）
    gc.collect()

    # 3. 清空CUDA缓存（释放未被使用的GPU内存）
    torch.cuda.empty_cache()

    # 初始化5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []  # 存储每一折的结果

    for fold, (train_indices, val_indices) in enumerate(kf.split(unique_key)):
        print(f"\n{'='*50}")
        print(f"开始第 {fold+1}/{n_splits} 折交叉验证")
        print(f"{'='*50}")

        # 划分当前折的训练集和验证集ID
        train_key = [unique_key[i] for i in train_indices]
        val_key = [unique_key[i] for i in val_indices]
        print(f"第{fold+1}折 - 训练样本数: {len(train_key)}, 验证样本数: {len(val_key)}")
        print("训练样本:",train_key)
        print("测试样本数:",val_key)

        train_dataset = generate_dataset1(
            train_key, all_question_trees, full_set_concepts,
            problem_feat_dict, concept_feat_dict, tree_dict
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1)

        val_dataset = generate_dataset2(
            val_key, all_question_trees, full_set_concepts,
            problem_feat_dict, concept_feat_dict, tree_dict
        )
        val_loader = DataLoader(val_dataset, batch_size=len(full_set_concepts)*n_sample, shuffle=False, collate_fn=collate_fn1)


        #构建模型（不再需要GNN，直接使用推理树文本的Qwen3编码作为树表征）
        classifier = SaMerClassifier(
            input_dim=3*dim,  # problem(dim) + concept(dim) + tree_text(dim)
            output_dim=classify_output_dim,
            num_hidden_layers=classify_num_hidden_layers,
            activation=classify_activation,
            dropout_rate=classify_dropout_rate
        ).to(device)
        # 多GPU并行：使用DataParallel在4张GPU上分布式训练分类器
        if torch.cuda.device_count() > 1:
            classifier = nn.DataParallel(classifier)
            print(f"  分类器已包装为DataParallel，使用 {torch.cuda.device_count()} 张GPU")
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate)
        criterion = nn.BCELoss()

        torch.cuda.empty_cache()  # 清理缓存
        torch.cuda.synchronize()  # 等待 GPU 操作完成，确保缓存清理生效

        # 训练循环
        print("开始训练模型")
        # 训练当前折
        fold_train_info = {}
        fold_val_info = {}
        max_val_f1 = 0.0
        best_val_epoch = 0

        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_tree_features = batch["tree_features"].to(device)
                batch_text_features = batch["text_features"].to(device)

                features = torch.cat([batch_text_features, batch_tree_features], dim=1)
                batch_labels = batch["labels"].to(device).float()
                batch_ids = batch['ids']

                outputs = classifier(features).squeeze()

                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            fold_train_info[epoch] = f"Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}")

            print("正在测试集上进行测试")

            # 验证
            classifier.eval()
            f1_list = {}

            with torch.no_grad():
                for batch in val_loader:
                    batch_tree_features = batch["tree_features"].to(device)
                    batch_text_features = batch["text_features"].to(device)

                    features = torch.cat([batch_text_features, batch_tree_features], dim=1)
                    batch_labels = batch["labels"].cpu().numpy()
                    batch_ids = batch['ids']
                    outputs = classifier(features).squeeze()

                    predicted = (outputs > 0.5).cpu().float().tolist()

                    # 计算每个id的F1分数
                    concept_count = len(full_set_concepts)
                    unique_ids = list(dict.fromkeys(batch_ids))
                    for i, id in enumerate(unique_ids):
                        start = i * concept_count
                        end = start + concept_count
                        f1 = f1_score(batch_labels[start:end], predicted[start:end], average='binary')
                        f1_list[id] = f1

                val_f1 = sum(f1_list.values())/len(f1_list)
            if epoch != num_epochs - 1:
                continue
            fold_val_info[epoch] = f"Val F1: {val_f1:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1} | Val F1: {val_f1:.4f}")
            all_epoch_results[epoch+1].append(val_f1)

            # 记录最佳验证准确率
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                best_val_epoch = epoch
                # 保存当前折的最佳模型（兼容DataParallel包装）
                model_state = classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict()
                # torch.save(model_state, f"../save_model/best_classifier1_tree_text.pth")

        # 记录当前折的结果
        fold_results.append({
            "fold": fold+1,
            "train_key": train_key,
            "val_key": val_key,
            "max_val_f1": max_val_f1,
            "best_epoch": best_val_epoch,
            "train_info": fold_train_info,
            "val_info": fold_val_info
        })
        print(f"第 {fold+1} 折最佳验证F1: {max_val_f1:.4f} (在第 {best_val_epoch+1} 轮)")

    # 汇总所有折的结果
    print("\n" + "="*50)
    print("5折交叉验证结果汇总")
    print("="*50)

    all_max_val_f1 = [fold["max_val_f1"] for fold in fold_results]
    print(f"各折最佳验证F1: {[f'{acc:.4f}' for acc in all_max_val_f1]}")
    print(f"平均验证F1: {np.mean(all_max_val_f1):.4f} ± {np.std(all_max_val_f1):.4f}")

    # 输出每折的详细信息
    for fold in fold_results:
        print(f"\n第 {fold['fold']} 折:")
        print(f"  最佳轮次: {fold['best_epoch']+1}")
        print(f"  最佳验证F1: {fold['max_val_f1']:.4f}")
        print(f"  训练样本数: {len(fold['train_key'])}")
        print(f"  验证样本数: {len(fold['val_key'])}")

    every_epoch_results = {}
    for a, b in all_epoch_results.items():
        every_epoch_results[a] = (np.mean(b),np.std(b, ddof=0))
    sorted_dict = sorted(every_epoch_results.items(), key=lambda x: x[1][0], reverse=True)
    print("平均epcosh的结果:",sorted_dict)
