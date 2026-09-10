import numpy as np
import json
from collections import Counter, defaultdict

# ==========================================
# 1. 读取数据
# ==========================================

results_path = "../output/analysis_regression_Algebra.json"
with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

# ==========================================
# 2. 核心分析逻辑
# ==========================================

def analyze_regression_errors(results):
    """
    分析难度预测（单标签5分类）的错误模式：
    1. 混淆矩阵 (5x5 Confusion Matrix)
    2. 各类别 Precision / Recall / F1
    3. 错误偏移分析 (Over-estimation vs Under-estimation)
    4. 错误距离分布 (相邻错误 vs 远端错误)
    """

    num_classes = 5
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'total': 0})
    error_distances = Counter()  # 预测错误的距离分布
    over_estimate = 0
    under_estimate = 0
    total_correct = 0
    total_samples = 0

    for qid, data in results.items():
        true_label = data['真实难度']
        pred_label = data['预测难度']

        confusion[true_label][pred_label] += 1
        total_samples += 1

        if true_label == pred_label:
            total_correct += 1
            class_stats[true_label]['TP'] += 1
        else:
            # 错误样本的偏移分析
            distance = abs(true_label - pred_label)
            error_distances[distance] += 1

            if pred_label > true_label:
                over_estimate += 1
            else:
                under_estimate += 1

            # 对真实类别来说是 FN，对预测类别来说是 FP
            class_stats[true_label]['FN'] += 1
            class_stats[pred_label]['FP'] += 1

        class_stats[true_label]['total'] += 1

    # ==========================================
    # 3. 输出分析结果
    # ==========================================

    # --- 分析 1: 混淆矩阵 ---
    print("=" * 55)
    print("1. 混淆矩阵 (行=真实难度, 列=预测难度)")
    print("=" * 55)
    header = "        " + "  ".join([f"Pred{i+1}" for i in range(num_classes)])
    print(header)
    for i in range(num_classes):
        row = f"True{i+1}  " + "  ".join([f"{confusion[i][j]:5d}" for j in range(num_classes)])
        print(row)

    # --- 分析 2: 各类别指标 ---
    print("\n" + "=" * 55)
    print("2. 各类别 Precision / Recall / F1")
    print("=" * 55)
    print(f"{'类别':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    all_precisions = []
    all_recalls = []
    all_f1s = []
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        tp = stats['TP']
        fp = stats['FP']
        fn = stats['FN']
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        print(f"难度{cls+1}  {precision:10.4f} {recall:10.4f} {f1:10.4f} {stats['total']:10d}")

    # 宏平均
    print(f"{'Macro Avg':<8} {np.mean(all_precisions):10.4f} {np.mean(all_recalls):10.4f} {np.mean(all_f1s):10.4f}")

    # --- 分析 3: 整体指标 ---
    print("\n" + "=" * 55)
    print("3. 整体指标")
    print("=" * 55)
    accuracy = total_correct / total_samples
    # WACC: 正确=1.0, 相邻错误=0.6
    wacc = total_correct
    for dist, count in error_distances.items():
        if dist == 1:
            wacc += count * 0.6
    wacc /= total_samples
    # MAE
    mae = sum(d * c for d, c in error_distances.items()) / total_samples
    print(f"  总样本数: {total_samples}")
    print(f"  正确数:   {total_correct}")
    print(f"  ACC:      {accuracy:.4f}")
    print(f"  WACC:     {wacc:.4f}")
    print(f"  MAE:      {mae:.4f}")

    # --- 分析 4: 错误偏移分析 ---
    print("\n" + "=" * 55)
    print("4. 错误偏移分析 (Over-estimation vs Under-estimation)")
    print("=" * 55)
    total_errors = over_estimate + under_estimate
    if total_errors > 0:
        print(f"  高估 (Pred > True): {over_estimate:4d}  ({over_estimate/total_errors*100:.1f}%)")
        print(f"  低估 (Pred < True): {under_estimate:4d}  ({under_estimate/total_errors*100:.1f}%)")
    else:
        print("  无错误样本")

    # --- 分析 5: 错误距离分布 ---
    print("\n" + "=" * 55)
    print("5. 错误距离分布 (预测与真实的绝对差值)")
    print("=" * 55)
    for dist in sorted(error_distances.keys()):
        count = error_distances[dist]
        print(f"  距离 {dist}: {count:4d} 次 ({count/total_errors*100:.1f}%)")

    # --- 分析 6: 每个难度类别的误差方向 ---
    print("\n" + "=" * 55)
    print("6. 每个难度类别的误差方向")
    print("=" * 55)
    class_error_direction = defaultdict(lambda: {'over': 0, 'under': 0, 'correct': 0})
    for qid, data in results.items():
        true_label = data['真实难度']
        pred_label = data['预测难度']
        if pred_label > true_label:
            class_error_direction[true_label]['over'] += 1
        elif pred_label < true_label:
            class_error_direction[true_label]['under'] += 1
        else:
            class_error_direction[true_label]['correct'] += 1

    for cls in sorted(class_error_direction.keys()):
        d = class_error_direction[cls]
        total = d['over'] + d['under'] + d['correct']
        print(f"  难度{cls+1}: 正确={d['correct']:3d}, 高估={d['over']:3d}, 低估={d['under']:3d}  (总计={total})")

    # --- 分析 7: 大误差样本 ---
    large_error_threshold = 2  # 预测与真实差距 >= 此值视为"差距过大"
    print("\n" + "=" * 55)
    print(f"7. 预测差距 >= {large_error_threshold} 的样本")
    print("=" * 55)
    large_error_samples = []
    for qid, data in results.items():
        true_label = data['真实难度']
        pred_label = data['预测难度']
        distance = abs(true_label - pred_label)
        if distance >= large_error_threshold:
            large_error_samples.append((qid, true_label, pred_label, distance))

    if large_error_samples:
        large_error_samples.sort(key=lambda x: -x[3])  # 按距离降序
        print(f"  共 {len(large_error_samples)} 个样本:")
        print(f"  {'ID':<10} {'真实难度':>8} {'预测难度':>8} {'差距':>6}")
        for qid, true_label, pred_label, distance in large_error_samples:
            print(f"  {qid:<10} {true_label+1:>8} {pred_label+1:>8} {distance:>6}")
    else:
        print(f"  无差距 >= {large_error_threshold} 的样本")

    return confusion, class_stats, error_distances


# ==========================================
# 4. 运行分析
# ==========================================
analyze_regression_errors(results)
