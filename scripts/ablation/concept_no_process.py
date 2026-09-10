import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import dgl
from dgl.nn import GINConv
import torch.nn.functional as F
from sklearn.model_selection import KFold  # 新增KFold导入
from collections import defaultdict
import gc

#--------------------------------------------------------------------------

# XE数据集的知识点预测任务，使用五折交叉方式，并优化验证流程提高计算效率，在classifier_train1_copy8_2的基础上，将构建表征放在五折循环外部
#使用DGL库
#不使用推理树，也不使用文本作答过程
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
def extract_features(model, tokenizer, texts, batch_size=10, max_length=400,use_tqdm=False):
    model.eval()
    features = []
    
    iterator = range(0, len(texts), batch_size)
    if use_tqdm:
        iterator = tqdm(iterator, desc="提取表征进度:")
    
    for i in iterator:
        #print_gpu_memory()
        #print_gpu_memory()
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
            batch_features = last_features.cpu()
        
        features.append(batch_features)
        del inputs, outputs, last_features  # 显式删除变量
        torch.cuda.empty_cache()  # 清理缓存
        torch.cuda.synchronize()  # 等待 GPU 操作完成，确保缓存清理生效
    
    return torch.cat(features)


def generate_dataset1(id_key, all_question_trees, full_set_concepts, problem_feat_dict, concept_feat_dict):
    features = []
    labels = []
    ids = []

    for id in tqdm(id_key, desc=f"处理进度:"):
        #print("id:",id)
        n = 0
        positive_concepts = all_question_trees[id]['concept']
        negative_concepts = [c for c in full_set_concepts if c not in positive_concepts]

        #for c in negative_concepts:
        for c in random.sample(negative_concepts, len(negative_concepts)//2):
            # 拼接问题特征和概念特征（直接从字典获取）
            feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
            features.append(feat)
            labels.append(0)
            ids.append(id)
            n += 1

        while n > 0:
            for c in positive_concepts:
                if n == 0:
                    break
                feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
                features.append(feat)
                labels.append(1)
                ids.append(id)
                n -= 1

    features = torch.stack(features)
    labels = torch.tensor(labels)
    
    # 创建数据集和数据加载器
    processed_dataset = ReasoningTreeDataset(text_features=features, labels=labels,ids=ids)

    return processed_dataset



def generate_dataset2(id_key, all_question_trees, full_set_concepts, problem_feat_dict, concept_feat_dict):
    features = []
    labels = []
    ids = []
    for id in tqdm(id_key, desc=f"构建验证集"):
        positive_concepts = all_question_trees[id]['concept']
        
        # 遍历所有概念（与原逻辑一致）
        for c in full_set_concepts:
            feat = torch.cat([problem_feat_dict[id], concept_feat_dict[c]], dim=0)
            features.append(feat)
            labels.append(1 if c in positive_concepts else 0)
            ids.append(id)
    # 转换为张量
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return ReasoningTreeDataset(text_features=features, labels=labels, ids=ids)

class ReasoningTreeNode:
    """推理树的节点类，用于表示题目、子问题及其推理过程"""
    def __init__(self, node_id, text=None, reasoning=None, parent=None):
        self.node_id = node_id      # 节点唯一标识
        self.text = text            # 题目/子问题文本
        self.reasoning = reasoning  # 对应的推理过程
        self.parent = parent        # 父节点
        self.children = []          # 子节点列表
    
    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)
        child_node.parent = self
        
    
#用于通过推理数据构建适用于图神经网络处理的数据结构
def ReasoningTree(tree_data,base_model,tokenizer):
    if len(tree_data) < 2:
        print("出现只有一个节点的树！！！直接推出程序")
        exit()
    
    nodes = []
    node_question_texts = []
    node_rationale_texts = []
    for id, branch in tree_data.items():
        if id == '0':
            #nodes.append(ReasoningTreeNode(int(id), branch['content'], branch['rationale']))
            nodes.append(ReasoningTreeNode(int(id)))
        else:
            #nodes.append(ReasoningTreeNode(int(id), branch['content'], branch['rationale'], branch['parent']))
            nodes.append(ReasoningTreeNode(int(id), None, None, branch['parent']))

        node_question_texts.append(branch['content'])
        node_rationale_texts.append(branch['rationale'])

    # 节点ID到索引的映射
    node_to_index =  {node.node_id: i for i, node in enumerate(nodes)}
    
    # 构建文本和推理的表征
    text_repr = extract_features(base_model,tokenizer,node_question_texts)
    reasoning_repr = extract_features(base_model,tokenizer,node_rationale_texts)
    # 拼接文本和推理的表征作为节点表征
    node_features = torch.cat([text_repr, reasoning_repr], dim=-1)
    #node_features = text_repr
    
    # # 添加节点层级作为节点属性
    # levels = []
    # level_map = {}
    # for j in range(len(nodes)):
    #     if j == 0:
    #         levels.append(0)
    #         level_map[0] = 0
    #     else:
    #         p = level_map[nodes[j].parent] + 1
    #         levels.append(p)
    #         level_map[nodes[j].node_id] = p
    # level_attr = torch.tensor(levels, dtype=torch.float).view(-1, 1)
    
    # 将层级属性与文本表征拼接
    # node_features = torch.cat([node_features, level_attr], dim=-1)
    
    # 构建边索引
    edges = []
    for node in nodes:
        if node.parent is not None:
            parent_idx = node_to_index[node.parent]
            child_idx = node_to_index[node.node_id]
            # 从父节点到子节点的边
            edges.append([parent_idx, child_idx])
            # 从子节点到父节点的边（双向连接）
            # edge_index.append([child_idx, parent_idx])
    
    # 创建DGL图
    if edges:  # 避免空图报错
        u, v = zip(*edges)
        g = dgl.graph((torch.tensor(u), torch.tensor(v)))
    else:
        g = dgl.graph(([], []), num_nodes=len(nodes))  # 空图
    
    # 存储节点特征到图中
    g.ndata['feat'] = node_features  # DGL图的节点特征存储在ndata中
    
    return {
        'graph': g,  # 返回DGL图
        'node_features': node_features,
        'batch': torch.zeros(node_features.size(0), dtype=torch.long),
        'num_nodes': node_features.size(0)
    }
    
class ReasoningTreeDataset(Dataset):
    def __init__(self, text_features, labels, ids):
        self.text_features= text_features
        self.labels = labels
        self.ids = ids
    
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        return {
            "text_feature": self.text_features[idx],  # 题目文本的原始表征
            "label": self.labels[idx],             # 难度值标签
            "id":self.ids[idx]
        }

class ReasoningTreeGNN(torch.nn.Module):
    """处理推理树的GNN模型，用于提取结构表征"""
    def __init__(self, input_dim, output_dim, graph_hidden_dim=24, dropout=0.1):
        super(ReasoningTreeGNN, self).__init__()

        self.graph_hidden_dim = graph_hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activate_fn = nn.ReLU()
        self.norm = nn.LayerNorm(self.graph_hidden_dim)  # 层归一化，稳定训练
        
        #节点特征投影（将原始高维特征映射到graph_hidden_dim）
        self.node_proj = nn.Linear(input_dim, graph_hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim)),
            GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim)),
            GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim)),
            GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim)),  # 新增第4层
            GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim))   # 新增第5层
        ])
        
        # # 三层GINConv（使用DGL的GINConv）
        # self.gnn_1 = GINConv(nn.Linear(graph_hidden_dim, graph_hidden_dim))
        # self.gnn_2 = GINConv(nn.Linear(graph_hidden_dim, graph_hidden_dim))
        # self.gnn_3 = GINConv(nn.Linear(graph_hidden_dim, graph_hidden_dim))
        

         # 图读出后的投影层（匹配原输出维度）
        self.out_proj = nn.Linear(graph_hidden_dim, output_dim)
    
    def forward(self, g, node_features):
        # 1. 节点特征投影到低维

        #print("node_features dtype:", node_features.dtype)
        # 打印 node_proj 权重的 dtype
        #print("node_proj weight dtype:", self.node_proj.weight.dtype)

        h = self.node_proj(node_features)

        for gnn in self.gnn_layers:
            # 残差连接：当前特征h与GIN输出相加
            h_res = h  # 保存输入特征
            h = gnn(g, h)  # GIN层计算
            h = h + h_res  # 残差相加（缓解梯度消失）
            
            h = self.activate_fn(h)  # 激活函数
            h = self.norm(h)  # 层归一化（稳定分布）
            h = self.dropout(h)  #  dropout（防止过拟合）
        
        # # 2. 三层GIN卷积
        # h = self.gnn_1(g, h)
        # h = self.activate_fn(h)
        # h = self.dropout(h)
        
        # h = self.gnn_2(g, h)
        # h = self.activate_fn(h)
        # h = self.dropout(h)
        
        # h = self.gnn_3(g, h)
        # h = self.activate_fn(h)
        
        # 3. 图全局读出（使用sum聚合）
        g.ndata['h'] = h
        g_readout = dgl.sum_nodes(g, 'h')  # 替换原global_mean_pool和max_pool
        
        # 4. 输出投影
        out = self.out_proj(g_readout)  
        return out     


class SaMerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_hidden_layers = 3,dropout_rate=0.2, activation=nn.SiLU):
        super(SaMerClassifier, self).__init__()
        
        #self.layer_sizes = [input_dim,1024,512,256,128]
        self.layer_sizes = [input_dim,1024,128]
        
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
    # 设置启动方式为spawn
    #mp.set_start_method('spawn', force=True)
    
    batch_size = 4000
    max_length = 500
    num_epochs = 70
    learning_rate = 2e-5
    seed = 46   #之前是46
    #分类模型参数
    #神经预测网络
    classify_num_hidden_layers = 0
    classify_activation=nn.SiLU
    classify_dropout_rate = 0.4
    classify_output_dim = 1
    #图神经网络参数
    gnn_layers = 3
    gnn_hidden_dim = 512
    gnn_type = 'gin'
    gnn_dropout = 0.4
    n_splits = 5  # 5折交叉验证
    n_sample = 20
    
    generate_model = True
    # 加载模型和Tokenizer
    device_map = {
    # 特殊模块集中分配
    "model.embed_tokens": 0,
    "model.norm": 0,
    "lm_head": 0,
    
    # 按分片文件分配主体层
    **{f"model.layers.{i}":  1 for i in range(48)}  # 所有层平分到每张卡
    }
    #model_path = "/inspire/hdd/project/ai4education/tongjunkai-p-tongjunkai/LLM/Qwen/Qwen3-Embedding-8B"
    #model_path = "/public1/home/stu52275901007/workspace/tjk_workspace/LLM/Qwen/Qwen3-Embedding-8B"
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
    base_model = AutoModel.from_pretrained(model_path,
                                            device_map='auto',
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True,
                                            attn_implementation="flash_attention_2")

    print("模型导入成功")
    #print("检查模型所有参数位置:")
    #for param in base_model.parameters():
        #print(param.device)  # 应看到参数分布在多个GPU上

    # 冻结主模型参数
    for param in base_model.parameters():
        param.requires_grad = False

    Dataset = args.dataset
    #Dataset_filepath = f'{args.data_dir}/{Dataset}/new_{Dataset}_trees3_application(include_concepts).json'
    Dataset_filepath = f'{args.data_dir}/{Dataset}/concept_items.json'
   
    
    with open(Dataset_filepath, 'r', encoding='utf-8') as f:
        all_question_trees = json.load(f)  # 假设这里读取的是顶层字典
    
    with open(f'{args.data_dir}/{Dataset}/questions全集中文版.json', 'r', encoding='utf-8') as f:
        origin_questions = json.load(f)  # 假设这里读取的是顶层字典
    
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
    #print("知识点全集:",full_set_concepts)
    
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
    solution_texts_all = []
    for id in unique_key:
        solu = origin_questions[id]['analysis'].replace('$','')
        solution_texts_all.append(solu)

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
            problem_feat_dict, concept_feat_dict
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = generate_dataset2(
            val_key, all_question_trees, full_set_concepts, 
            problem_feat_dict, concept_feat_dict
        )
        val_loader = DataLoader(val_dataset, batch_size=len(full_set_concepts)*n_sample, shuffle=False)
        
        #构建模型
        classifier = SaMerClassifier(3*dim).to(device)

        classifier = SaMerClassifier(
            input_dim=2*dim,
            output_dim=classify_output_dim,
            num_hidden_layers=classify_num_hidden_layers,
            activation=classify_activation,
            dropout_rate=classify_dropout_rate
        ).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
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
                #处理样本
                batch_text_features = batch["text_feature"].to(device).float()
                features = batch_text_features
                
                batch_labels = batch["label"].to(device).float()
                batch_ids = batch['id']
                #print("ids:",batch_ids)

                outputs = classifier(features).squeeze()
                #print("outputs:",outputs)

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
            #print_gpu_memory()
            f1_list = {}
            
            with torch.no_grad():  # 修正2：添加no_grad以加速推理
                for batch in val_loader:
                    #处理样本
                    batch_text_features = batch["text_feature"].to(device).float()
                    
                    features = batch_text_features
                    batch_labels = batch["label"].cpu().numpy()
                    batch_ids = batch['id']
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
                    
                #print("f1列表:",f1_list)
                val_f1 = sum(f1_list.values())/len(f1_list) 
            # if epoch != num_epochs - 1:
            #     continue
            fold_val_info[epoch] = f"Val F1: {val_f1:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1} | Val F1: {val_f1:.4f}")
            all_epoch_results[epoch+1].append(val_f1)
            
            # 记录最佳验证准确率
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                best_val_epoch = epoch
                # 保存当前折的最佳模型
                torch.save(classifier.state_dict(), f"../save_model/best_classifier1.pth")
                
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

    
    

