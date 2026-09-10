import os
# ============ GPU配置（修改此处设置使用的GPU） ============
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 如 "0"=单卡，"0,1"=双卡，"0,1,2,3"=四卡
# =========================================================

import json
from tqdm import tqdm
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 本地模型配置（按需修改） ============
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "/inspire/hdd/project/ai4education/qianhong-p-qianhong/LLM/Qwen3-8B")  # 模型路径或HuggingFace模型名，如 "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 256                # 最大生成token数
MAX_INPUT_LENGTH = 8192             # 每个样本最大输入长度（token数），超长部分会被截断；设为 None 则不限制
ENABLE_THINKING = False             # Qwen3 思考模式：True=开启，False=关闭（关闭后不再输出 <think> 块）
# ===============================================

temperature = 1.0
top_p = 0.7
max_retry = 10    
batch_size = 100

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
    texts = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if MAX_INPUT_LENGTH is not None:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)
    else:
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
    """提取最后一个匹配的数字（防止LLM输出多个数值）"""
    close_tag = tag[:1] + '/' + tag[1:]  # </tag>
    pattern = re.escape(tag) + r'(-?\d+)' + re.escape(close_tag)
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[-1])
    else:
        raise ValueError(f"无法从 '{text}' 中提取数字")


def compute_true_label(id_a, id_b, item_params):
    """根据IRT beta计算真实标签"""
    beta_a = item_params[id_a]['beta']
    beta_b = item_params[id_b]['beta']
    diff = beta_a - beta_b
    if diff > 0.5:
        return 1, beta_a, beta_b, diff
    elif diff < -0.5:
        return -1, beta_a, beta_b, diff
    else:
        return 0, beta_a, beta_b, diff


if __name__ == "__main__":

    Dataset = "XES-1500"
    model_name = LOCAL_MODEL_PATH.split("/")[-1]  # 自动从模型路径提取模型名

    Data_path = f"../../data/{Dataset}/difficulty_items.json"
    item_params_path = f"../../data/{Dataset}/irt_parameters.json"
    judge_path = f"../../output/LLMonTree_Difficulty_PAIR_{model_name}_{Dataset}.json"

    prompt = """Judge which of the following two questions, question a or question b, is more difficult based on the text of the questions and their reasoning trees.
If their difficulties are about the same, output 0; If question a is more difficult, output 1; If question b is more difficult, output -1.

[question a]:
{}

[reasoning tree of question a]:
{}

[question b]:
{}

[reasoning tree of question b]:
{}

Instruction:
1. Please strictly output in the following format: <tag>0/1/-1</tag>. '0/1/-1' is used to indicate which of the two questions is more difficult.
2. Please provide only the final difficulty label. Do not output any additional information.
"""

    folds = {
        'fold1':['1106', '763', '1041', '3128', '312', '3233', '3196', '3467', '580', '660', '1470', '1313', '328', '1740', '813', '659', '439', '369', '843', '1001', '2022', '708', '3210', '2402', '611', '546', '232', '2033', '1414', '240', '1305', '1884', '1027', '482', '2962', '665', '248', '3047', '1133', '1395', '1247', '3173', '3204', '726', '171', '768', '489', '502', '554', '622', '64', '2140', '1641', '939', '3561', '3528', '3512', '686', '798', '3345', '1243', '2100', '937', '147', '2193', '1264', '690', '999', '2763', '305', '884', '651', '1173', '205', '2269', '1582', '241', '1183', '341', '249', '1528', '42', '169', '672', '3003', '1341', '1259', '775', '771', '2160', '1295', '2035', '1032', '148', '676', '99', '474', '116', '426', '122', '2827', '259', '2654', '1330', '1585', '731', '2683', '257', '1480', '2197', '2042', '1000', '1407', '2854', '3024', '12', '702', '1181', '2579', '3501', '821', '2721', '440', '91', '1152', '3199', '7', '1333', '3183', '1153', '2130', '174', '729', '101', '1503', '74', '97', '673', '1806', '1687', '3536', '1498', '480', '1155', '3007', '920', '2117', '1756', '2906', '1331', '1334', '176', '3553', '1191', '2142', '2747', '3232', '724', '2917', '475', '1390', '103', '3108', '246', '1244', '3230', '327', '2785', '559', '683', '1871', '1500', '141', '397', '530', '1067', '2313', '1512', '3594', '1156', '1301', '2665', '1185', '876', '3296', '1704', '1024', '613', '649', '3076', '1297', '764', '0', '2308', '556', '1217', '563', '2963', '258', '949', '560', '1239', '3160', '3105', '923', '3198', '945', '2026', '238', '942', '958', '239', '1590'],
        'fold2':['1995', '175', '3197', '557', '2529', '54', '1298', '1141', '1308', '170', '860', '52', '837', '215', '960', '2899', '2826', '3191', '861', '3042', '2949', '3287', '1975', '1257', '1888', '1828', '760', '603', '2288', '1046', '470', '472', '3527', '132', '653', '2825', '712', '1031', '1158', '1343', '1002', '912', '2959', '1', '230', '605', '503', '3120', '41', '1337', '2878', '304', '477', '143', '1118', '1389', '1339', '329', '374', '2929', '33', '529', '3176', '1446', '3158', '535', '2429', '133', '2107', '1698', '1270', '2431', '2583', '2841', '479', '540', '655', '2991', '2732', '117', '1945', '728', '2966', '1275', '2778', '2426', '36', '1388', '3063', '1955', '1293', '1599', '1008', '332', '585', '1218', '855', '3111', '538', '1240', '606', '1653', '830', '1210', '1342', '1492', '998', '995', '599', '1520', '3074', '1215', '2069', '1393', '700', '893', '2304', '808', '1304', '1299', '1162', '86', '163', '43', '1547', '3167', '796', '1279', '422', '687', '317', '1048', '840', '1071', '149', '614', '208', '1783', '552', '16', '880', '857', '1573', '233', '2616', '991', '3187', '497', '3172', '1921', '94', '113', '89', '3044', '79', '226', '751', '1340', '1065', '1242', '1450', '3468', '1755', '2759', '2974', '2108', '723', '102', '839', '938', '1261', '1154', '1846', '2526', '72', '1809', '162', '139', '438', '3194', '144', '118', '3395', '753', '346', '55', '217', '3092', '401', '3129', '909', '1300', '969', '3100', '3281', '549', '997', '2334', '882', '2341', '1850', '1190', '2232', '1650', '3002', '2614', '1574', '125', '1289', '1437', '180', '1941', '2139'],
        'fold3':['3178', '865', '666', '3321', '704', '1078', '35', '473', '1207', '929', '334', '2374', '667', '504', '547', '3202', '193', '1132', '486', '1805', '3231', '805', '427', '680', '604', '1335', '484', '722', '894', '3241', '566', '1592', '3043', '2062', '2276', '1063', '1105', '3148', '553', '897', '31', '1177', '797', '883', '1025', '1303', '1290', '1856', '1346', '202', '3235', '251', '179', '710', '1184', '2422', '1121', '2077', '568', '1128', '1045', '83', '1159', '2901', '1973', '1142', '2714', '1488', '940', '1817', '1061', '578', '732', '3083', '3039', '1956', '1517', '213', '2896', '1314', '1174', '2105', '1603', '375', '693', '40', '924', '2287', '2856', '705', '902', '2291', '866', '75', '2617', '644', '1049', '2135', '677', '2375', '1584', '1952', '2293', '138', '675', '1186', '2292', '92', '1733', '1245', '78', '1263', '1047', '2666', '1518', '992', '3258', '1003', '2396', '1349', '544', '2277', '1404', '1483', '120', '1675', '142', '1262', '950', '2017', '1678', '642', '476', '256', '2984', '1981', '1951', '1859', '485', '2286', '1140', '1172', '761', '1535', '507', '366', '2980', '300', '1499', '3067', '654', '2012', '1565', '1451', '558', '891', '498', '469', '2740', '2086', '1280', '108', '886', '204', '247', '1702', '350', '2230', '888', '555', '1248', '1294', '1651', '490', '682', '3162', '1157', '1077', '671', '1265', '1212', '1487', '2978', '1062', '2029', '678', '56', '3137', '2428', '319', '609', '1050', '1188', '1074', '2338', '769', '647', '793', '46', '253', '2074', '2577', '365', '1695', '2939', '1348', '1219', '1915', '1591', '1803', '2397', '898', '936'],
        'fold4': ['1161', '1070', '607', '1274', '650', '2824', '827', '1268', '1654', '579', '2087', '858', '727', '565', '320', '2020', '1066', '1456', '3374', '1208', '61', '901', '3469', '2789', '2774', '664', '899', '2695', '2952', '3064', '934', '652', '288', '1475', '646', '123', '759', '3030', '879', '1272', '721', '922', '725', '2432', '1396', '1307', '707', '2926', '1735', '368', '153', '85', '1392', '250', '2028', '2194', '543', '1127', '2021', '209', '1958', '598', '87', '1444', '1241', '2038', '168', '150', '155', '1292', '2391', '51', '3132', '488', '140', '684', '3012', '326', '145', '762', '98', '2393', '1104', '919', '770', '2761', '3502', '527', '670', '237', '1076', '468', '134', '508', '1760', '623', '2133', '772', '1216', '703', '820', '110', '321', '1258', '3398', '39', '895', '45', '1033', '586', '887', '2716', '1214', '2746', '822', '1187', '3157', '38', '1302', '2612', '441', '173', '755', '767', '539', '757', '3270', '3591', '1510', '695', '617', '1178', '2686', '561', '562', '2036', '57', '455', '311', '754', '874', '1160', '131', '2427', '941', '1291', '1143', '688', '1117', '1073', '859', '842', '3221', '1189', '2255', '96', '943', '1347', '2620', '255', '3569', '542', '733', '2158', '3302', '3079', '1588', '662', '1098', '342', '1246', '1040', '3211', '2195', '119', '2394', '786', '281', '681', '73', '3333', '3596', '506', '908', '927', '245', '685', '467', '3027', '2937', '487', '3428', '790', '3303', '3134', '1842', '718', '2295', '1013', '1923', '1471', '1700', '2071', '564', '701', '1054', '3583', '3136', '836', '699', '3195', '3273'],
        'fold5':['1583', '244', '2285', '1278', '62', '434', '458', '3190', '854', '1060', '90', '178', '916', '1957', '765', '3396', '2736', '2040', '1079', '838', '417', '658', '528', '669', '3547', '2290', '1804', '231', '1909', '656', '1338', '791', '1418', '88', '913', '3238', '689', '1522', '2068', '3430', '1594', '1030', '709', '889', '1614', '1917', '1151', '1537', '71', '773', '1277', '325', '835', '1267', '425', '844', '545', '228', '1182', '3040', '1513', '587', '1916', '333', '2916', '483', '853', '1326', '616', '3257', '3548', '2967', '3159', '643', '2167', '3103', '928', '993', '3584', '1505', '911', '2234', '1568', '663', '37', '532', '900', '2728', '95', '3006', '353', '1742', '1147', '2063', '717', '430', '2777', '1394', '2693', '2466', '534', '2085', '424', '165', '1681', '167', '1401', '600', '2938', '610', '2667', '1391', '537', '2309', '3175', '59', '881', '3253', '533', '124', '3261', '2034', '1504', '3085', '242', '1005', '2645', '2592', '679', '2474', '648', '2724', '1064', '1655', '3334', '758', '1310', '1175', '2078', '926', '3515', '2229', '877', '657', '608', '1075', '130', '1296', '2794', '1209', '536', '1336', '1638', '318', '2470', '3075', '996', '2749', '1402', '2256', '593', '3431', '1213', '551', '1211', '621', '1288', '198', '3521', '990', '826', '1068', '80', '2289', '584', '625', '624', '137', '135', '550', '875', '3516', '2198', '674', '243', '661', '1954', '100', '3117', '2384', '845', '1306', '146', '136', '1578', '2261', '601', '2072', '3510', '1220', '414', '2739', '105', '618', '591', '2253', '428', '3082', '3560', '915', '668', '1122']
             }

    # ============ 对每一折构建去重的两两样本对 ============
    fold_pairs = {}
    for fold_name, sample_ids in folds.items():
        pairs = []
        n = len(sample_ids)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((sample_ids[i], sample_ids[j]))
        fold_pairs[fold_name] = pairs
        print(f"{fold_name}: {n} 样本 -> {len(pairs)} 样本对")
    # =====================================================

    with open(Data_path, 'r', encoding='utf-8') as f:
        Data = json.load(f)

    with open(item_params_path, 'r', encoding='utf-8') as f:
        item_params = json.load(f)

    # ============ 数据格式校验 ============
    data_keys_sample = list(Data.keys())[:3]
    params_keys_sample = list(item_params.keys())[:3]
    all_fold_ids = [id for fold_ids in folds.values() for id in fold_ids]
    fold_id_sample = all_fold_ids[:3]
    print(f"Data 前3个key: {data_keys_sample} (类型: {type(data_keys_sample[0]).__name__ if data_keys_sample else 'N/A'})")
    print(f"item_params 前3个key: {params_keys_sample} (类型: {type(params_keys_sample[0]).__name__ if params_keys_sample else 'N/A'})")
    print(f"fold 前3个ID: {fold_id_sample} (类型: {type(fold_id_sample[0]).__name__ if fold_id_sample else 'N/A'})")
    missing_data = [id for id in all_fold_ids if id not in Data]
    missing_params = [id for id in all_fold_ids if id not in item_params]
    n_total = len(set(all_fold_ids))
    print(f"唯一fold样本数: {n_total}, Data缺失: {len(missing_data)}, item_params缺失: {len(missing_params)}")
    # ===================================

    if os.path.exists(judge_path):
        with open(judge_path, 'r', encoding='utf-8') as f:
            all_judge = json.load(f)
        print("目标文件存在")
    else:
        all_judge = {}
        print("目标文件不存在")

    # ============ 兼容旧嵌套格式：若为 {fold_name: {pair_key: result}} 则迁移为扁平 {pair_key: result} ============
    if all_judge and any(k.startswith('fold') and isinstance(v, dict) for k, v in all_judge.items()):
        new_judge = {}
        for k, v in all_judge.items():
            if isinstance(v, dict):
                for pair_k, pair_val in v.items():
                    new_judge[pair_k] = pair_val
            else:
                new_judge[k] = v
        all_judge = new_judge
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(all_judge, f, ensure_ascii=False, indent=2)
        print("已将旧嵌套格式迁移至扁平格式并保存")
    # ========================================================================================================

    fold_names = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    fold_results = {}

    for fold_name in fold_names:
        pairs = fold_pairs[fold_name]

        # 筛选未处理的样本对（扁平结构，直接检查 f"{id_a}|{id_b}" 是否已在 all_judge 中）
        remain_pairs = [(a, b) for (a, b) in pairs if f"{a}|{b}" not in all_judge]

        if not remain_pairs:
            fold_total = 0
            fold_correct = 0
            for id_a, id_b in pairs:
                key = f"{id_a}|{id_b}"
                if key in all_judge:
                    fold_total += 1
                    if all_judge[key]['pred'] == all_judge[key]['true']:
                        fold_correct += 1
            fold_acc = fold_correct / fold_total if fold_total > 0 else 0
            fold_results[fold_name] = {'acc': fold_acc, 'total': fold_total, 'correct': fold_correct}
            print(f"\n{fold_name} 已全部处理完成，ACC: {fold_acc:.4f} ({fold_correct}/{fold_total})")
            continue

        print(f"\n===== {fold_name}: 待处理 {len(remain_pairs)} 对 =====")

        # 过滤有效样本对
        valid_pairs = []
        for id_a, id_b in remain_pairs:
            if id_a not in Data or id_b not in Data:
                print(f"警告: ({id_a}, {id_b}) 数据不存在，跳过")
                continue
            if id_a not in item_params or id_b not in item_params:
                print(f"警告: ({id_a}, {id_b}) IRT参数不存在，跳过")
                continue
            valid_pairs.append((id_a, id_b))

        skipped_data = len(remain_pairs) - len(valid_pairs)
        if skipped_data > 0:
            print(f"因数据缺失跳过 {skipped_data} 对，有效样本对: {len(valid_pairs)}")
        if not valid_pairs:
            print(f"[警告] {fold_name} 无有效样本对，跳过本折")
            continue
        for chunk_start in tqdm(range(0, len(valid_pairs), batch_size), desc=fold_name):
            chunk_pairs = valid_pairs[chunk_start:chunk_start + batch_size]

            # 构建当前批次的prompt和元数据
            chunk_contents = []
            chunk_meta = []
            for id_a, id_b in chunk_pairs:
                true_label, beta_a, beta_b, diff = compute_true_label(id_a, id_b, item_params)
                content = prompt.format(
                    Data[id_a]['question'], Data[id_a]['reasoning_tree'],
                    Data[id_b]['question'], Data[id_b]['reasoning_tree']
                )
                chunk_contents.append(content)
                chunk_meta.append((id_a, id_b, true_label, beta_a, beta_b, diff))

            # 一次 model.generate() 处理整批
            responses = batch_llm_inference(chunk_contents)

            for (id_a, id_b, true_label, beta_a, beta_b, diff), response in zip(chunk_meta, responses):
                if response is None:
                    continue
                try:
                    response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                    pred_label = extract_number_regex(response_clean)
                    print(f"预测标签: {pred_label}, 真实标签: {true_label} (beta_a={beta_a:.3f}, beta_b={beta_b:.3f}, diff={diff:.3f})")
                    all_judge[f"{id_a}|{id_b}"] = {'pred': pred_label, 'true': true_label}
                except Exception as e:
                    print(f"[解析失败] ({id_a}|{id_b}) 模型输出无法提取标签: {e}")
                    print(f"  原始输出: {response[:200]}")

            # 每个batch结束后保存
            with open(judge_path, 'w', encoding='utf-8') as f:
                json.dump(all_judge, f, ensure_ascii=False, indent=2)

        # 每折处理完保存
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(all_judge, f, ensure_ascii=False, indent=2)

        # 计算该折ACC（从扁平 all_judge 中按 fold_pairs 查找）
        fold_total = 0
        fold_correct = 0
        for id_a, id_b in pairs:
            key = f"{id_a}|{id_b}"
            if key in all_judge:
                fold_total += 1
                if all_judge[key]['pred'] == all_judge[key]['true']:
                    fold_correct += 1
        fold_acc = fold_correct / fold_total if fold_total > 0 else 0
        fold_results[fold_name] = {'acc': fold_acc, 'total': fold_total, 'correct': fold_correct}
        print(f"{fold_name} ACC: {fold_acc:.4f} ({fold_correct}/{fold_total})")

    # 总体结果（直接从扁平 all_judge 里筛选所有 pair 条目统计）
    pair_entries = {k: v for k, v in all_judge.items() if isinstance(v, dict) and 'pred' in v and 'true' in v}
    total_pairs = len(pair_entries)
    total_correct = sum(1 for v in pair_entries.values() if v['pred'] == v['true'])
    overall_acc = total_correct / total_pairs if total_pairs > 0 else 0

    print("\n\n----------------------------------------------------------------")
    print(f"总样本对数: {total_pairs}")
    print(f"总体ACC: {overall_acc:.4f} ({total_correct}/{total_pairs})")
    for fold_name in fold_names:
        r = fold_results[fold_name]
        print(f"  {fold_name}: ACC = {r['acc']:.4f} ({r['correct']}/{r['total']})")

    all_judge['总体ACC'] = overall_acc
    all_judge['各折ACC'] = {fn: fold_results[fn]['acc'] for fn in fold_names}
    with open(judge_path, 'w', encoding='utf-8') as f:
        json.dump(all_judge, f, ensure_ascii=False, indent=2)
    print("结果文件已保存至：", judge_path)
