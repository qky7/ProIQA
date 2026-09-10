import numpy as np
import pandas as pd
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt
#import seaborn as sns
import json

# ==========================================
# 1. 模拟数据 (请替换为你真实的结果字典)
# ==========================================
# 假设数据格式： {题目ID: {'true': [真实标签列表], 'pred': [预测标签列表]}}


# ==========================================
# 2. 核心分析逻辑
# ==========================================

def analyze_error_patterns(results):
    """
    分析知识点预测的错误模式：
    1. 混淆矩阵 (Top-N Confusion Pairs)
    2. 频次偏差 (Head/Tail Performance)
    3. 标签数量分析 (Cardinality/Missing Secondary)
    """
    
    # --- 初始化统计变量 ---
    confusion_pairs = []  # 存储 (True_Label, Pred_Label) 的错误对
    label_counts = Counter()  # 统计真实标签出现次数 (用于区分 Head/Tail)
    
    # 用于计算每个标签的 TP, FP, FN (针对频次分析)
    label_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    # 用于计算不同标签数量下的准确率 (针对次要知识点分析)
    cardinality_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'subset_acc': 0})

    print("-" * 50)
    print("🚀 开始分析错误模式...")
    print("-" * 50)

    # --- 遍历所有样本 ---
    for qid, data in results.items():
        true_labels = set(data['真实知识点'])
        pred_labels = set(data['预测知识点'])
        
        # 更新真实标签计数
        label_counts.update(true_labels)
        
        # --- A. 混淆分析逻辑 ---
        # 找出 Miss (漏测) 和 Hallucination (误报)
        missed = true_labels - pred_labels
        hallucinated = pred_labels - true_labels
        
        # 如果既有漏测又有误报，可能存在混淆
        # 简单策略：将所有 Miss 和 Hallucination 两两组合作为潜在混淆对
        # (更复杂的策略可以使用语义相似度，但在无额外信息下，全排列是合理的近似)
        for m in missed:
            for h in hallucinated:
                confusion_pairs.append((m, h))  # (真实, 预测错误)
        
        # --- B. 频次分析逻辑 (TP/FP/FN) ---
        # TP: 预测对的
        tp = true_labels & pred_labels
        for label in tp:
            label_metrics[label]['TP'] += 1
        
        # FN: 漏测的
        for label in missed:
            label_metrics[label]['FN'] += 1
            
        # FP: 误报的
        for label in hallucinated:
            label_metrics[label]['FP'] += 1
            
        # --- C. 标签数量分析逻辑 ---
        num_labels = len(true_labels)
        cardinality_stats[num_labels]['total'] += 1
        
        # 计算 Subset Accuracy (完全匹配)
        if true_labels == pred_labels:
            cardinality_stats[num_labels]['subset_acc'] += 1
            
        # 计算 Recall (召回率) - 只要预测对了一部分也算
        if len(true_labels) > 0:
            recall = len(tp) / len(true_labels)
            cardinality_stats[num_labels]['correct'] += recall  # 累加 Recall

    # ==========================================
    # 3. 输出分析结果
    # ==========================================

    # --- 分析 1: 易混淆知识点 (Confusing Related Concepts) ---
    print("\n📊 1. Top-10 易混淆知识点对 (True -> Pred):")
    confusion_counts = Counter(confusion_pairs)
    for (true_lbl, pred_lbl), count in confusion_counts.most_common(10):
        print(f"   - 真实: {true_lbl} -> 被误判为: {pred_lbl} (共 {count} 次)")
    
    # --- 分析 2: 频次偏差分析 (Head vs Tail) ---
    print("\n📊 2. 频次偏差分析 (Head vs Tail Performance):")
    # 将标签按频次排序
    sorted_labels = label_counts.most_common()
    num_labels = len(sorted_labels)
    
    # 定义 Head (前20%), Body (中间), Tail (后20%)
    head_cutoff = int(num_labels * 0.2)
    tail_cutoff = int(num_labels * 0.8)
    
    groups = {
        'Head (High Freq)': sorted_labels[:head_cutoff],
        'Body (Mid Freq)': sorted_labels[head_cutoff:tail_cutoff],
        'Tail (Low Freq)': sorted_labels[tail_cutoff:]
    }
    
    for group_name, labels in groups.items():
        total_tp, total_fp, total_fn = 0, 0, 0
        for lbl, _ in labels:
            m = label_metrics[lbl]
            total_tp += m['TP']
            total_fp += m['FP']
            total_fn += m['FN']
            
        # 计算该组的微平均 F1 (Micro-F1)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        print(f"   - {group_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # --- 分析 3: 标签数量分析 (Missing Secondary Concepts) ---
    print("\n📊 3. 复杂题目分析 (Performance by Number of Labels):")
    # 按标签数量排序输出
    for num in sorted(cardinality_stats.keys()):
        stats = cardinality_stats[num]
        avg_recall = stats['correct'] / stats['total']
        subset_acc = stats['subset_acc'] / stats['total']
        print(f"   - {num} 个知识点: 平均Recall={avg_recall:.3f}, 完全匹配率={subset_acc:.3f} (样本数: {stats['total']})")

    return confusion_counts, groups, cardinality_stats

# ==========================================
# 4. 运行分析
# ==========================================
# 请将 results 替换为你真实的数据字典

results_path = "../output/analysis_concept_XES-500.json"
with open(results_path, 'r', encoding='utf-8') as f:  
    results = json.load(f)
analyze_error_patterns(results)