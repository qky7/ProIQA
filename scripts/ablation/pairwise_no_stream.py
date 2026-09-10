import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import dgl
from dgl.nn import GINConv
from torch import Tensor
from tqdm import tqdm 
import random
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
import json
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from sklearn.model_selection import KFold  # 新增KFold导入
from collections import defaultdict

#--------------------------------------------------------------------------

# XE数据集的难度预测任务，使用五折交叉方式。将GNN中的GIN层改为cao lang的线形层
#完全采用caolang的图神经网络
#参考家乐的代码进行修改
#没有推理树和文本作答过程
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
            #计算最后一个有效token的表征
            outputs = model(**inputs)
            last_features = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            batch_features = last_features.cpu()
            
        features.append(batch_features)
    
    return torch.cat(features)

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

def collate_fn1(batch):
    """
    自定义数据拼接函数，处理不同节点数量的推理树
    """
    trees = [item["tree"] for item in batch]
    text_features = torch.stack([item["text_feature"] for item in batch]).to(device)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float).to(device)
    ids = [item['id'] for item in batch]
    
     # 处理DGL图批处理（核心修改）
    graphs = [tree["graph"] for tree in trees]
    batched_graph = dgl.batch(graphs)  # DGL批量处理图
    node_features = torch.cat([tree["node_features"] for tree in trees], dim=0).to(device)
    
    return {
        "graphs": batched_graph,  # 批处理后的DGL图
        "node_features": node_features,  # 拼接的节点特征
        "text_features": text_features,
        "labels": labels,
        "ids": ids
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
            # GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim)),  # 新增第4层
            # GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim))   # 新增第5层
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
  #难度预测模型
class SaMerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_hidden_layers = 3,dropout_rate=0.2, activation=nn.SiLU):
        super(SaMerClassifier, self).__init__()
        
        #self.layer_sizes = [input_dim,1024,512,256,128]
        #self.layer_sizes = [input_dim,1024,512,512,512,128]
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
    
def get_ranking_indices(difficulty_sequence):
    """
    计算难度序列中每个元素的排序序号（从0开始）
    
    参数:
        difficulty_sequence: 包含难度值的列表
        
    返回:
        每个元素对应的排序序号列表
    """
    # 对难度序列进行排序，得到排序后的列表
    sorted_sequence = sorted(difficulty_sequence)
    
    # 对于原序列中的每个元素，找到它在排序后列表中的索引位置
    ranking_indices = [sorted_sequence.index(num) for num in difficulty_sequence]
    
    return ranking_indices


device = "cuda" if torch.cuda.is_available() else "cpu"

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
    seed = 48
    train_batch_size = 55
    val_batch_size = 55
    max_length = 500
    num_epochs = 50
    learning_rate = 4e-5
    threshold = 0.5    #构造训练样本标签时的阈值
    gamma = 0.3 #训练中的阈值
    margin = 1  #训练时防止梯度爆炸
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
    #排序损失和回归损失设置
    pair_loss_rate = 0.8
    regress_loss_rate = 0.2
    # 交叉验证参数
    n_splits = 5  # 5折交叉验证
    
    model_path = args.model_path
    # 加载模型和Tokenizer
    base_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,     #也可以torch.float16
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

    # 冻结主模型参数
    for param in base_model.parameters():
        param.requires_grad = False
    
    Dataset = 'XES-1500'
    # 步骤1: 读取原始JSON文件

    with open(f'{args.data_dir}/{Dataset}/difficulty_items.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)  # 假设这里读取的是顶层字典
    #with open(f'{args.data_dir}/{Dataset}/{Dataset}_trees4.json', 'r', encoding='utf-8') as f:
    #    questions = json.load(f)  # 假设这里读取的是顶层字典
        
    with open(f"{args.data_dir}/{Dataset}/irt_parameters.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    
    print("questions长度:",len(questions))
    print("items长度:",len(items))
    
    if len(questions) != 500:
        print('原始数据元素数量不匹配，直接推出')
        #exit()
        
    # 步骤2: 提取并处理键（可选择是否随机打乱）
    unique_key = list(questions.keys())
    unique_key.sort()
    print("清洗前，样本总数:",len(unique_key))
    
    #去除推理树有问题的样本
    to_remove = []
    for id in unique_key:
        if len(questions[id]['reasoning_tree']) < 2 or questions[id]['reasoning_tree']["0"]["rationale"] == "":
            to_remove.append(id)
    for id in to_remove:
        unique_key.remove(id)
    print("需删除的样本:",to_remove)

    print("清洗后，总样本数:",len(unique_key))
    
    # 分割训练集和测试集
    # 打乱数据顺序
    # 设置随机种子
    set_seed(seed)
    random.shuffle(unique_key)

    print("处理数据集")
    #构建训练数据集    
    all_problem_texts = []
    all_labels = {}

    all_epoch_acc_results = defaultdict(list)
    
    for id in tqdm(unique_key):
        all_problem_texts.append(questions[id]['question'])
        #all_labels[id] = items[id]['beta'] #难度值预测
        all_labels[id] = items[id]['beta'] #难度值预测
        
    problem_text_features = extract_features(base_model, tokenizer, all_problem_texts)
    all_text_features = {id: problem_text_features[i] for i, id in enumerate(unique_key)}    
    
    # 初始化5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []  # 存储每一折的结果
    
     # 开始交叉验证
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
        
         # 准备训练数据
        train_text_features = torch.stack([all_text_features[id] for id in train_key])
        train_labels = [all_labels[id] for id in train_key]
        train_ids = train_key.copy()
        
        # 准备验证数据
        val_text_features = torch.stack([all_text_features[id] for id in val_key])
        val_labels = [all_labels[id] for id in val_key]
        val_ids = val_key.copy()
    
        # 创建数据集和数据加载器
        train_dataset = ReasoningTreeDataset(
                                            text_features=train_text_features, 
                                            labels=train_labels,
                                            ids = train_ids
                                            )
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

        # 创建数据集和数据加载器
        val_dataset = ReasoningTreeDataset(
                                            text_features=val_text_features, 
                                            labels=val_labels,
                                            ids = val_ids
                                            )
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)
    
    
        # 初始化分类头
        # 初始化模型（每个fold重新初始化）
        classifier = SaMerClassifier(
            input_dim=base_model.config.hidden_size,
            output_dim=classify_output_dim,
            num_hidden_layers=classify_num_hidden_layers,
            activation=classify_activation,
            dropout_rate=classify_dropout_rate
        ).to(device)
        
        optimizer = torch.optim.Adam(list(classifier.parameters()), lr=learning_rate)
    
        #np.set_printoptions(suppress=True, precision=5)
    
        # 训练当前折
        fold_train_info = {}
        fold_val_info = {}
        max_val_acc = 0.0
        max_val_sp = 0
        best_val_acc_epoch = 0
        best_val_sp_epoch = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            all_train_preds = []# 存储所有样本的预测难度值
            all_train_trues = []# 存储所有样本的真实难度值
            all_train_ids = []
                    
            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):

                batch_text_features = batch["text_feature"].to(device)
                features = batch_text_features.to(torch.float) 
                
                batch_labels = batch["label"].to(device).float()
                batch_ids = batch['id']
                
                # 前向传播
                pred = classifier(features).squeeze(-1)
                
                # 收集当前batch的预测值和真实值
                with torch.no_grad():
                    all_train_preds.extend(pred.detach().cpu().numpy())
                    all_train_trues.extend(batch_labels.cpu().numpy())
                    all_train_ids.extend(batch_ids)
                
                # 生成基础索引
                batch_size = len(batch_ids)
                #print("batch_size:",batch_size)
                
                idx = np.arange(batch_size)
                
                # 生成网格索引（类似torch.meshgrid）
                idx_i, idx_j = np.meshgrid(idx, idx, indexing='ij')  # 注意指定indexing='ij'保持一致

                # 创建掩码（排除i=j的情况）
                mask = idx_i < idx_j
                
                # 筛选有效索引对
                idx_i = idx_i[mask]  # 展平为一维数组，只保留i≠j的元素
                idx_j = idx_j[mask]
                
                # 通过索引获取样本对的预测值（无需重复前向传播）
                pred_i = pred[idx_i]  # 样本i的预测值
                pred_j = pred[idx_j]  # 样本j的预测值
                
                labels_i = batch_labels[idx_i]
                labels_j = batch_labels[idx_j]

                true_diff = labels_i - labels_j
                pair_true_labels = torch.where(
                    true_diff > threshold, 1.0,
                    torch.where(true_diff < -1*threshold, -1.0, 0.0)
                )
                
                # 计算排序损失
                delta = pred_i - pred_j
                hinge_term = torch.abs(pair_true_labels) * torch.clamp(-pair_true_labels * delta + gamma, min=0)
                distance_term = (1 - torch.abs(pair_true_labels)) * torch.clamp(torch.abs(delta), max=margin)  # 限制最大距离
                ranking_loss = (hinge_term + distance_term).mean()
                
                #计算回归损失
                regression_loss = (torch.nn.functional.mse_loss(pred_i, labels_i) + torch.nn.functional.mse_loss(pred_j, labels_j)) / 2
                
                loss = pair_loss_rate*ranking_loss + regress_loss_rate*regression_loss    
                #loss = regression_loss    
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                with torch.no_grad():
                    # 计算差值并生成标签
                    pred_diff = pred_i - pred_j
                    pair_pred_labels = torch.where(
                        pred_diff > threshold, 1.0,
                        torch.where(pred_diff < -1*threshold, -1.0, 0.0)
                    )
                    
                    correct += (pair_pred_labels == pair_true_labels).sum().item()
                    total += len(pair_pred_labels)
                
            # 将列表转换为numpy数组（确保一一对应）
            all_train_preds = np.array(all_train_preds)
            all_train_trues = np.array(all_train_trues)
            all_train_ids = np.array(all_train_ids)
            # 直接使用斯皮尔曼相关
            train_sp, train_p_value = stats.spearmanr(all_train_preds, all_train_trues)
                    
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            fold_train_info[epoch] = f"Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, SP: {train_sp:.4f}, P: {train_p_value:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | ACC: {train_acc:.4f}, SP: {train_sp:.4f} | P: {train_p_value:.4f}")
            
            if False:
                print(f"训练集题目序号排序:{all_train_ids}")
                print(f"真实难度序列:{all_train_trues}")
                print(f"真实排序:{get_ranking_indices(all_train_trues)}")
                print(f"预测难度序列:{all_train_preds}")
                print(f"预测排序:{get_ranking_indices(all_train_preds)}")
            
            # 根据训练损失保存最佳模型
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     torch.save(classifier.state_dict(), "../save_model/best_regression_regressor.pth")
            #     torch.save(GNNModel.state_dict(),"../save_model/best_regression_GNN.pth")
            #     print("!!!!!!Saved best model!!!!!!")
        
            # 验证
            classifier.eval()
            val_total = 0
            val_correct = 0
            cnt = 0
            all_val_preds = []# 存储所有样本的预测难度值
            all_val_trues = []# 存储所有样本的真实难度值
            all_val_ids = []
            
            with torch.no_grad():
                for batch in val_loader:
                    
                    batch_text_features = batch["text_feature"].to(device)
                    
                    features = batch_text_features.to(torch.float) 

                    batch_labels = batch["label"].to(device).float()
                    batch_ids = batch['id']
                    
                    # 前向传播
                    pred = classifier(features).squeeze(-1)
                    
                    all_val_preds.extend(pred.detach().cpu().numpy())
                    all_val_trues.extend(batch_labels.cpu().numpy())
                    all_val_ids.extend(batch_ids)
                    
                    # 生成基础索引
                    batch_size = len(batch_ids)
                    #print("batch_size:",batch_size)
                    idx = np.arange(batch_size)
                    
                    # 生成网格索引（类似torch.meshgrid）
                    idx_i, idx_j = np.meshgrid(idx, idx, indexing='ij')  # 注意指定indexing='ij'保持一致

                    # 创建掩码（排除i=j的情况）
                    mask = idx_i < idx_j
                    
                    # 筛选有效索引对
                    idx_i = idx_i[mask]  # 展平为一维数组，只保留i≠j的元素
                    idx_j = idx_j[mask]
                    
                     # 新增：随机抽取1/50的样本对
                    num_pairs = len(idx_i)
                    if num_pairs == 0:
                        continue  # 避免无样本对时出错
                    # 计算抽样数量（1/50），确保至少抽取1个
                    sample_size = max(1, num_pairs // 10)
                    # 随机选择样本对索引
                    np.random.seed(seed)  # 可选：固定种子保证结果可复现
                    random_indices = np.random.choice(num_pairs, size=sample_size, replace=False)
                    # 只保留抽样后的样本对
                    idx_i = idx_i[random_indices]
                    idx_j = idx_j[random_indices]
                    
                    if epoch == 0:
                        print(f"fold {fold+1} 验证集上的采样序号")
                        print("采样的序号i:",idx_i)
                        print("采样的序号j:",idx_j)
                    
                    # 通过索引获取样本对的预测值（无需重复前向传播）
                    pred_i = pred[idx_i]  # 样本i的预测值，形状: [num_pairs]
                    pred_j = pred[idx_j]  # 样本j的预测值，形状: [num_pairs]
                    
                    labels_i = batch_labels[idx_i]
                    labels_j = batch_labels[idx_j]

                    true_diff = labels_i - labels_j
                    pair_true_labels = torch.where(
                        true_diff > threshold, 1.0,
                        torch.where(true_diff < -1*threshold, -1.0, 0.0)
                    )
                    
                    # 计算差值并生成标签
                    pred_diff = pred_i - pred_j
                    pair_pred_labels = torch.where(
                        pred_diff > threshold, 1.0,
                        torch.where(pred_diff < -1*threshold, -1.0, 0.0)
                    )
                    
                    val_correct += (pair_pred_labels == pair_true_labels).sum().item()
                    val_total += len(pair_pred_labels)
                    
                    if False:
                        for x in range(len(idx_i)):
                            site_i = batch_ids[idx_i[x]]
                            site_j = batch_ids[idx_j[x]]
                            if (site_i,site_j) not in test_fine_analysis.keys():
                                test_fine_analysis[(site_i,site_j)] = {'pre_label':[0,0,0],'true_label':int(pair_true_labels[x].item())}
                            lab = int(pair_pred_labels[x].item())
                            #print("lab:",lab)
                            test_fine_analysis[(site_i,site_j)]['pre_label'][lab+1] += 1
            
            
            # 将列表转换为numpy数组（确保一一对应）
            all_val_preds = np.array(all_val_preds)
            all_val_trues = np.array(all_val_trues)
            all_val_ids = np.array(all_val_ids)
                
            # 直接使用斯皮尔曼相关
            val_sp, val_p_value = stats.spearmanr(all_val_preds, all_val_trues)
                        
            val_acc = val_correct / val_total

            # if epoch != num_epochs-1:
            #     continue
            
            # # ---------------------- 新增：计算预测排序和真实排序 ----------------------
            # # 1. 计算预测值的升序排名（从1开始，值越小排名越前）
            # # argsort返回升序排序后的原始索引（例如：preds=[3,1,2] → argsort结果为[1,2,0]）
            # pred_sorted_indices = np.argsort(all_val_preds)
            # # 初始化预测排名数组，按升序索引赋值排名（1-based）
            # pred_ranks = np.zeros_like(all_val_preds, dtype=int)
            # for rank, idx in enumerate(pred_sorted_indices, start=1):
            #     pred_ranks[idx] = rank  # 第idx个样本的预测排名为rank

            # # 2. 计算真实值的升序排名（逻辑同上）
            # true_sorted_indices = np.argsort(all_val_trues)
            # true_ranks = np.zeros_like(all_val_trues, dtype=int)
            # for rank, idx in enumerate(true_sorted_indices, start=1):
            #     true_ranks[idx] = rank  # 第idx个样本的真实排名为rank\

            # # 3. 整合每个样本的id、预测值、真实值、预测排名、真实排名
            # sample_rank_info = []
            # for i in range(len(all_val_ids)):
            #     # sample_info = {
            #     #     "id": all_val_ids[i],          # 样本id
            #     #     "pred_value": all_val_preds[i], # 预测值
            #     #     "true_value": all_val_trues[i], # 真实值
            #     #     "pred_rank": pred_ranks[i],     # 预测升序排名（1-based）
            #     #     "true_rank": true_ranks[i]      # 真实升序排名（1-based）
            #     # }
            #     # sample_rank_info.append(sample_info)
            #     print(f"样本id:{all_val_ids[i]},预测值:{ all_val_preds[i]},预测排序:{pred_ranks[i]},真实值:{all_val_trues[i]},真实排序:{true_ranks[i]}")

            fold_val_info[epoch] = f"Val ACC: {val_acc:.4f}, SP: {val_sp:.4f}, P: {val_p_value:.4f}"
            print(f"Fold {fold+1} Epoch {epoch+1} | Val ACC: {val_acc:.4f} | SP: {val_sp:.4f}, P: {val_p_value:.4f}")
            all_epoch_acc_results[epoch+1].append(val_acc)

            # 记录最佳验证准确率
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_val_acc_epoch = epoch
                # 保存当前折的最佳模型
                #torch.save(classifier.state_dict(), f"../save_model/best_regression_regressor.pth")
                #torch.save(GNNModel.state_dict(), f"../save_model/best_regression_GNN.pth")
                
            if val_sp > max_val_sp:
                max_val_sp = val_sp
                best_val_sp_epoch = epoch
                    
        # 记录当前折的结果
        fold_results.append({
            "fold": fold+1,
            "train_key": train_key,
            "val_key": val_key,
            "max_val_acc": max_val_acc,
            "best_acc_epoch": best_val_acc_epoch,
            "max_val_sp": max_val_sp,
            "best_sp_epoch": best_val_sp_epoch,
            "train_info": fold_train_info,
            "val_info": fold_val_info
        })
        print(f"第 {fold+1} 折, 最佳验证ACC: {max_val_acc:.4f} (在第 {best_val_acc_epoch+1} 轮) | 最佳验证SP: {max_val_sp:.4f} (在第 {best_val_sp_epoch+1} 轮)")
    
     # 汇总所有折的结果
    print("\n" + "="*50)
    print("5折交叉验证结果汇总")
    print("="*50)
    
    all_max_val_acc = [fold["max_val_acc"] for fold in fold_results]
    print(f"各折最佳验证ACC: {[f'{acc:.4f}' for acc in all_max_val_acc]}")
    print(f"平均验证ACC: {np.mean(all_max_val_acc):.4f} ± {np.std(all_max_val_acc):.4f}")
    
    all_max_val_sp = [fold["max_val_sp"] for fold in fold_results]
    print(f"各折最佳验证SP: {[f'{sp:.4f}' for sp in all_max_val_sp]}")
    print(f"平均验证SP: {np.mean(all_max_val_sp):.4f} ± {np.std(all_max_val_sp):.4f}")
    
    # 输出每折的详细信息
    for fold in fold_results:
        print(f"\n第 {fold['fold']} 折:")
        print(f"  最佳验证ACC轮次: {fold['best_acc_epoch']+1}")
        print(f"  最佳验证ACC: {fold['max_val_acc']:.4f}")
        print(f"  最佳验证SP轮次: {fold['best_sp_epoch']+1}")
        print(f"  最佳验证SP: {fold['max_val_sp']:.4f}")
        print(f"  训练样本数: {len(fold['train_key'])}")
        print(f"  验证样本数: {len(fold['val_key'])}")   


    every_epoch_acc_results = {}
    for a, b in all_epoch_acc_results.items():
        every_epoch_acc_results[a] = (np.mean(b),np.std(b, ddof=0))
    sorted_acc_dict = sorted(every_epoch_acc_results.items(), key=lambda x: x[1][0], reverse=True)
    print("平均epcosh的ACC结果:",sorted_acc_dict)
                
    
        
        
        
    

