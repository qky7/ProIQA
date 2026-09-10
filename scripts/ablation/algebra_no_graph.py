import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import random
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
import json
from torch.utils.data import DataLoader, Dataset
import re
from collections import defaultdict
from sklearn.model_selection import KFold  # 新增KFold导入

#-----------------------------------------------------------------------------

# MATH数据集的难度预测任务，使用五折交叉方式
# 修改版本：将推理树文本整体传入Qwen3-8B提取隐藏状态，替代原有的DGL图神经网络表征构建方式

#-----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
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

# 特征提取函数
def extract_features(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    features = []

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
            outputs = model(**inputs)
            #计算最后一个有效token的表征
            last_features = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            batch_features = last_features.cpu().float()

        features.append(batch_features)

    return torch.cat(features)


def collate_fn1(batch):
    """
    自定义数据拼接函数，处理推理树文本表征（不再使用DGL图）
    """
    tree_features = torch.stack([item["tree"] for item in batch]).to(device)
    text_features = torch.stack([item["text_feature"] for item in batch]).to(device)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long).to(device)
    ids = [item['id'] for item in batch]

    return {
        "tree_features": tree_features,  # 推理树文本的Qwen3编码表征
        "text_features": text_features,
        "labels": labels,
        "ids": ids
    }

class ReasoningTreeDataset(Dataset):
    def __init__(self, trees, text_features, labels,ids):
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


#难度预测模型
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = 46
    train_batch_size = 55
    test_batch_size = 55
    max_length = 500
    tree_max_length = 1024  # 推理树文本最大长度
    num_epochs = 50
    learning_rate = 1e-4
    #分类模型参数
    classify_num_hidden_layers = 0
    classify_activation=nn.SiLU
    classify_dropout_rate = 0.4
    classify_output_dim = 5
    # 交叉验证参数
    n_splits = 5  # 5折交叉验证

    model_path = args.model_path
    # 加载模型和Tokenizer
    base_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,     #也可以torch.float16
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

    # 冻结主模型参数
    for param in base_model.parameters():
        param.requires_grad = False

    Dataset = 'Algebra'
    # 步骤1: 读取原始JSON文件

    with open(f'{args.data_dir}/{Dataset}/difficulty_items.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)  # 假设这里读取的是顶层字典

    with open(f"{args.data_dir}/{Dataset}/items.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    print("questions长度:",len(questions))
    print("items长度:",len(items))

    if len(questions) != 500 or len(items) != 500:
        print('原始数据元素数量不匹配，直接推出')
        #exit()

    # 步骤2: 提取并处理键（可选择是否随机打乱）
    unique_key = list(questions.keys())
    print("清洗前，样本总数:",len(unique_key))

    unique_key.sort()

    #去除推理树有问题的样本
    if True:
        to_remove = []
        for id in unique_key:
            if  questions[id]['reasoning_tree'] == None or len(questions[id]['reasoning_tree']) < 2 or questions[id]['reasoning_tree']["0"]["rationale"] == "":
                to_remove.append(id)
        print("需删去的样本:",to_remove)
        for id in to_remove:
            unique_key.remove(id)
        print("清洗后，总样本数:",len(unique_key))

    # 分割训练集和测试集
    # 打乱数据顺序
    # 设置随机种子
    set_seed(seed)

    unique_key = np.array(unique_key)
    np.random.shuffle(unique_key)
    unique_key = unique_key.tolist()

    all_label_distribute = {0:0,1:0,2:0,3:0,4:0}

    #构建训练数据集
    all_text_features = {}
    all_problem_texts = []
    all_labels = {}
    all_reprocessed_trees = {}

    all_epoch_acc_results = defaultdict(list)
    all_epoch_wacc_results = defaultdict(list)
    all_epoch_pred_label = {}

    for id in tqdm(unique_key):
        all_problem_texts.append(questions[id]['question'])
        lab = int(re.findall(r'\d+', items[id]['level'])[0])-1
        all_labels[id] = lab
        all_label_distribute[lab] += 1

    print("数据集标签分布:",all_label_distribute)

    problem_text_features = extract_features(base_model, tokenizer, all_problem_texts)
    all_text_features = {id: problem_text_features[i] for i, id in enumerate(unique_key)}

    # 格式化并编码所有推理树文本
    print("预计算所有推理树特征（文本编码方式）...")
    tree_texts = []
    for id in unique_key:
        tree_text = format_reasoning_tree_text(questions[id]['reasoning_tree'])
        tree_texts.append(tree_text)
    tree_features = extract_features(base_model, tokenizer, tree_texts, max_length=tree_max_length)
    all_reprocessed_trees = {id: tree_features[i] for i, id in enumerate(unique_key)}


    # 初始化5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []  # 存储每一折的结果
     # 开始交叉验证
    for fold, (train_indices, val_indices) in enumerate(kf.split(unique_key)):
        set_seed(seed)
        print(f"\n{'='*50}")
        print(f"开始第 {fold+1}/{n_splits} 折交叉验证")
        print(f"{'='*50}")

        # 划分当前折的训练集和验证集ID
        train_key = [unique_key[i] for i in train_indices]
        val_key = [unique_key[i] for i in val_indices]
        print(f"第{fold+1}折 - 训练样本数: {len(train_key)}, 验证样本数: {len(val_key)}")
        print("训练样本:",train_key)
        print("测试样本数:",val_key)

         # 准备训练数据
        train_trees = [all_reprocessed_trees[id] for id in train_key]
        train_text_features = torch.stack([all_text_features[id] for id in train_key])
        train_labels = [all_labels[id] for id in train_key]

        # 准备验证数据
        val_trees = [all_reprocessed_trees[id] for id in val_key]
        val_text_features = torch.stack([all_text_features[id] for id in val_key])
        val_labels = [all_labels[id] for id in val_key]

        # 创建数据集和数据加载器
        train_dataset = ReasoningTreeDataset(
            trees=train_trees,
            text_features=train_text_features,
            labels=train_labels,
            ids=train_key
        )
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn1,num_workers=0)

        val_dataset = ReasoningTreeDataset(
            trees=val_trees,
            text_features=val_text_features,
            labels=val_labels,
            ids=val_key
        )
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn1,num_workers=0)

        # 初始化模型（不再需要GNN，直接使用推理树文本的Qwen3编码作为树表征）
        classifier = SaMerClassifier(
            input_dim=2*base_model.config.hidden_size,  # problem(dim) + tree_text(dim)
            output_dim=classify_output_dim,
            num_hidden_layers=classify_num_hidden_layers,
            activation=classify_activation,
            dropout_rate=classify_dropout_rate
        ).to(device)

        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        # 训练当前折
        fold_train_info = {}
        fold_val_info = {}
        max_val_acc = 0.0
        max_val_wacc = 0
        best_val_acc_epoch = 0
        best_val_wacc_epoch = 0

        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
                batch_tree_features = batch["tree_features"].to(device)
                batch_text_features = batch["text_features"].to(device)

                features = torch.cat([batch_text_features, batch_tree_features], dim=1)

                batch_labels = batch["labels"].to(device)

                # 前向传播
                outputs = classifier(features)

                loss = criterion(outputs, batch_labels)

                #计算分类损失
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*batch_labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            fold_train_info[epoch] = f"Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}")


            # 验证
            classifier.eval()
            val_correct = 0
            val_total = 0
            val_wacc = 0
            val_true_labels = {}
            val_pred_labels = {}
            with torch.no_grad():
                for batch in val_loader:
                    batch_tree_features = batch["tree_features"].to(device)
                    batch_text_features = batch["text_features"].to(device)

                    features = torch.cat([batch_text_features, batch_tree_features], dim=1)

                    batch_labels = batch["labels"].to(device)
                    batch_ids = batch["ids"]

                    # 前向传播
                    outputs = classifier(features)

                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

                    for i in range(len(preds)):
                        if preds[i].item() == batch_labels[i].item():
                            val_wacc += 1
                        elif abs(preds[i].item() - batch_labels[i].item()) == 1:
                            val_wacc += 0.6

            if epoch != num_epochs-1:
                continue
            val_acc = val_correct / val_total
            val_wacc = val_wacc / val_total

            fold_val_info[epoch] = f"Val ACC: {val_acc:.4f}, Val Wacc: {val_wacc:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1} | Val ACC: {val_acc:.4f} | Val Wacc: {val_wacc:.4f}")
            all_epoch_acc_results[epoch+1].append(val_acc)
            all_epoch_wacc_results[epoch+1].append(val_wacc)

            # 记录最佳验证准确率
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_val_acc_epoch = epoch

            if val_wacc > max_val_wacc:
                max_val_wacc = val_wacc
                best_val_wacc_epoch = epoch

        # 记录当前折的结果
        fold_results.append({
            "fold": fold+1,
            "train_key": train_key,
            "val_key": val_key,
            "max_val_acc": max_val_acc,
            "best_acc_epoch": best_val_acc_epoch,
            "max_val_wacc": max_val_wacc,
            "best_wacc_epoch": best_val_wacc_epoch,
            "train_info": fold_train_info,
            "val_info": fold_val_info
        })
        print(f"第 {fold+1} 折, 最佳验证ACC: {max_val_acc:.4f} (在第 {best_val_acc_epoch+1} 轮) | 最佳验证WACC: {max_val_wacc:.4f} (在第 {best_val_wacc_epoch+1} 轮)")


    # 汇总所有折的结果
    print("\n" + "="*50)
    print("5折交叉验证结果汇总")
    print("="*50)

    all_max_val_acc = [fold["max_val_acc"] for fold in fold_results]
    print(f"各折最佳验证ACC: {[f'{acc:.4f}' for acc in all_max_val_acc]}")
    print(f"平均验证ACC: {np.mean(all_max_val_acc):.4f} ± {np.std(all_max_val_acc):.4f}")

    all_max_val_wacc = [fold["max_val_wacc"] for fold in fold_results]
    print(f"各折最佳验证WACC: {[f'{wacc:.4f}' for wacc in all_max_val_wacc]}")
    print(f"平均验证WACC: {np.mean(all_max_val_wacc):.4f} ± {np.std(all_max_val_wacc):.4f}")

    # 输出每折的详细信息
    for fold in fold_results:
        print(f"\n第 {fold['fold']} 折:")
        print(f"  最佳验证ACC轮次: {fold['best_acc_epoch']+1}")
        print(f"  最佳验证ACC: {fold['max_val_acc']:.4f}")
        print(f"  最佳验证WACC轮次: {fold['best_wacc_epoch']+1}")
        print(f"  最佳验证WACC: {fold['max_val_wacc']:.4f}")
        print(f"  训练样本数: {len(fold['train_key'])}")
        print(f"  验证样本数: {len(fold['val_key'])}")

    every_epoch_acc_results = {}
    for a, b in all_epoch_acc_results.items():
        every_epoch_acc_results[a] = (np.mean(b),np.std(b, ddof=0))
    sorted_acc_dict = sorted(every_epoch_acc_results.items(), key=lambda x: x[1][0], reverse=True)
    print("平均epcosh的ACC结果:",sorted_acc_dict)

    every_epoch_wacc_results = {}
    for a, b in all_epoch_wacc_results.items():
        every_epoch_wacc_results[a] = (np.mean(b),np.std(b, ddof=0))
    sorted_wacc_dict = sorted(every_epoch_wacc_results.items(), key=lambda x: x[1][0], reverse=True)
    print("平均epcosh的WACC结果:",sorted_wacc_dict)

    print("每折上选定的epoch的验证集数据标签预测情况：",all_epoch_pred_label)
