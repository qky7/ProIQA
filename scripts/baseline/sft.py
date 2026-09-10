#-------------------------------------------------------------------------
#该代码用于训练SFT的LLM
#------------------------------------------------------------------------
from datasets import Dataset
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import random
import numpy as np

PaddingID = -100

template1 = """
    请你详细浏览并理解下方所提供的题目文本、作答过程。
    通过题目文本和作答过程，判断该题的题目难度，题目难度范围0-1.请只输出题目难度值，不要输出额外信息。
    题目文本:{}
    作答过程:{}
    """

def preprocess_dataset(key, questions, all_question_solutions, question_acc,tokenizer, max_len=500, overflow_strategy='truncate'):
    """
    @function:预处理输入数据
    examples：数据集
    max_len：尽管 Qwen2-7B-Instruct 支持 131072 tokens，但最好不要设置为最大长度，否则显存占用将会非常大。
            max_len 的设置可以通过统计数据集的 token 长度得到。具体方法：将所有数据输入到 qwen2 模型的 tokenizer，统计 tokenizer 的输出长度（最大，最小，平均）
    overflow_strategy：'drop'表示丢弃，'truncate'表示截断
    """

    model_inputs = {'input_ids': [], 'labels': [], 'input_len': [], 'output_len': []}
    for id in key:
        
        message = [
            {"role": "system", "content": "你是一个知识渊博的人，请根据问题做出全面且正确的回答。"},
            {"role": "user", "content": template1.format(questions[id]['content'],all_question_solutions[id])},
        ]
        prompt = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=False,  # 不添加生成提示符
                tokenize=False                # 返回字符串用于调试
            )
        a_ids = tokenizer.encode(prompt)
        b_ids = tokenizer.encode(f"{question_acc[id]}", add_special_tokens=False) + [tokenizer.eos_token_id]
        context_length = len(a_ids)
        input_ids = a_ids + b_ids

        if len(input_ids) > max_len and overflow_strategy == 'drop':
            # 丢弃样本
            input_ids = []
            labels = []
        else:
            if max_len > len(input_ids):
                #使用 -100 填充, 因为 torch.nn.CrossEntropyLoss 的 ignore_index=-100, 即 CrossEntropyLoss 会忽略标签为 -100 的值的 loss，只计算非填充部分的 loss
                #torch.nn.CrossEntropyLoss 官方文档：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                pad_length = max_len - len(input_ids)
                labels = [PaddingID] * context_length + b_ids + [PaddingID] * pad_length
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            else:
                # 超过最大长度的数据被截断
                labels = [PaddingID] * context_length + b_ids
                labels = labels[:max_len]
                input_ids = input_ids[:max_len]
        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['input_len'].append(len(a_ids))
        model_inputs['output_len'].append(len(b_ids))
    return model_inputs

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


# 设置可见GPU（物理GPU0-3）
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 

# 设备映射规则（逻辑设备0-3）
device_map = {
    # 特殊模块集中分配
    "model.embed_tokens": 0,
    "model.norm": 0,
    "lm_head": 0,
    
    # 按分片文件分配主体层
    **{f"model.layers.{i}": i//5 for i in range(36)}  # 所有层平分到每张卡
}

max_memory = {
    0: "40GB",  # 承载前12层+embedding
    1: "40GB",
    2: "40GB",
    3: "40GB"   # 最后12层+norm+head
}

if __name__=="__main__":
    seed = 43
    # load tokenizer
    model_name = 'Qwen3-Embedding'
    model_path = os.environ.get("MODEL_PATH", "/inspire/hdd/global_user/limingjia-p-limingjia/LLM/Qwen3-Embedding-8B")
    #model_path = "/inspire/hdd/global_user/limingjia-p-limingjia/LLM/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16, 
                                                device_map=device_map, 
                                                #max_memory=max_memory,
                                                trust_remote_code=True,
                                                attn_implementation="flash_attention_2"
                                                )
    #print("输出gpu分配")
    #for name, param in model.named_parameters():
    #    print(f"{name}: {param.device}")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # load training dataset
    Dataset_name = "XES3G5M"
    with open(f"../../data/{Dataset_name}/questions.json", "r", encoding="utf-8") as file:
        questions = json.load(file)  # 直接返回 Python
    with open(f"../../data/{Dataset_name}/all_question_solution.json", "r", encoding="utf-8") as file:
        all_question_solutions = json.load(file)  # 直接返回 Python
    with open(f"../../data/{Dataset_name}/question_acc.json", "r", encoding="utf-8") as file:
        question_acc = json.load(file)  # 直接返回 Python
    
    # 步骤2: 提取并处理键（可选择是否随机打乱）
    unique_key = set(questions.keys()) & set(all_question_solutions.keys()) & set(question_acc.keys())
    unique_key = list(unique_key)
    unique_key.sort()
    print("样本总数:",len(unique_key))
    
    # 分割训练集和测试集
    # 打乱数据顺序
    # 设置随机种子
    set_seed(seed)
    random.shuffle(unique_key)

    # 计算分割点（80%训练集，20%测试集）
    split_index = int(0.8 * len(unique_key))
    train_key = unique_key[:split_index]
    test_key = unique_key[split_index:]    
    print("训练样本数:",len(train_key))
    print("测试上本数:",len(test_key))
    
    train_data_dict = preprocess_dataset(train_key, questions, all_question_solutions, question_acc,tokenizer)
    train_dataset = Dataset.from_dict(train_data_dict)

    test_data_dict = preprocess_dataset(test_key, questions, all_question_solutions, question_acc,tokenizer)
    test_dataset = Dataset.from_dict(test_data_dict)
    
    """
    加载 train.xxx 文件,如 train.txt  train.jsonl
    batched：分批加载数据，默认 batch=1000
    num_proc：配置多线程处理，一般不设置单线程的数据加载速度也很快
    load_from_cache_file：指定是否从缓存文件加载预处理后的数据。如果设置为 True，datasets 库会尝试从磁盘加载预先处理并缓存的数据集，而不是重新运行 map 函数。
                        设置 load_from_cache_file=False 意味着每次运行脚本时都会重新进行数据预处理，而不是从缓存中加载。
    """

    sft_model_path = os.path.join('../../save_model/SFT_Model',model_name)
    if not os.path.exists(sft_model_path):
        os.mkdir(sft_model_path)
        print(f"文件夹 {sft_model_path} 创建成功！")
    else:
        print(f"文件夹 {sft_model_path} 已存在。")

    """
    启用模型的梯度检查点, 梯度检查点是一种优化技术，可用于减少训练时的内存消耗。
    在反向传播期间，模型的中间激活值需要被保留以计算梯度。
    梯度检查点技术通过仅保存必要的一部分激活值，并在需要时重新计算丢弃的激活值，从而减少内存使用。
    """

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                        model=model, 
                                        label_pad_token_id=PaddingID, 
                                        pad_to_multiple_of=None, 
                                        padding=False)
    """
    创建了一个 Seq2Seq任务 的数据整理器, 用于将多个样本组合成一个批次。
    label_pad_token_id：指定用于填充标签的 padding token 的 id, 默认为-100
    pad_to_multiple_of = None：指定padding后序列长度应该是多少的倍数。如果设置为None（默认值），则不进行这种类型的padding。
    padding = False：指定是否对数据进行padding。设置为False 通常意味着数据的 padding 将在模型内部或通过其他方式处理。
    """
    
    # 训练参数
    args = TrainingArguments(
        output_dir=f'../../save_model/SFT_Model/{model_name}',             # 模型保存路径
        logging_dir='../../logs',          # 日志保存路径
        logging_strategy='steps',           # 按步骤记录
        per_device_train_batch_size=2,      # 用于指定训练的每个GPU/CPU的batch，每个训练步骤中每个硬件上的样本数量。
        logging_steps=1,
        gradient_accumulation_steps=10,     # 梯度累计，在显存较小的设备中，每隔多个 batch_size 更新一次梯度；
                                     # 真正更新梯度的 batch = per_device_train_batch_size * gradient_accumulation_steps
                                            # 即 4*32=128 个 batch 更新一次梯度
        gradient_checkpointing=True,  # 启用梯度检查点（减少显存占用）
        num_train_epochs=5,                 # sft llm 的 epoch 一般不需要太大，1～3轮即可
        weight_decay=0.003,                 # 权重衰减正则化，将一个与权重向量的L2范数成比例的惩罚项加到总损失中
        warmup_ratio=0.03,                  # 预热，在训练初期逐渐增加学习率，而不是从一开始就使用预设的最大学习率，避免一开始就使用过高的学习率可能导致的训练不稳定。
                                            # 如果设置 warmup_ratio=0.1，共有100个epochs，那么在前10个epochs（即前10%的训练时间），学习率会从0逐渐增加到最大值。
        optim='adafactor',                # adafactor或 adamw_torch
        lr_scheduler_type="cosine",         # 根据余弦函数的形状来逐渐减小学习率，一般有 "linear" 和 "cosine" 两种方式   
        learning_rate=1e-5,                 # 最大学习率
        eval_strategy="steps",  # 新增评估策略
        eval_steps=30,                # 与保存步长一致
        save_strategy='steps',
        save_steps=30,                       # 保存模型的步骤，save_steps 是 per_device_train_batch_size * gradient_accumulation_steps，而不是 per_device_train_batch_size
        save_total_limit=3, # 关键修改：只保留x个检查点 
        load_best_model_at_end=True,    # 关键修改：加载最佳模型
        metric_for_best_model="eval_loss",  # 新增评估指标
        greater_is_better=False,    # 损失值越小越好
        #bf16=if_bf16,                       # 是否使用 bfloat16 数据格式
        bf16=True,
        run_name=model_name,
        report_to='none',                  # 使用 wandb 打印日志
    )
    
    print(f"训练使用的GPU数量: {args.world_size}")
    
    # 检查实际使用的GPU数量
    if torch.cuda.is_available():
        print(f"系统可见GPU数量: {torch.cuda.device_count()}")  # 应为8
        print(f"当前进程使用的GPU: {torch.cuda.current_device()}")

    # train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
