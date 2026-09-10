<h2 align="center">
ProIQA: A Process-Based Framework for Fine-Grained Math Item Quality Assessment
</h2>

<h3 align="center">
IEEE ICDM 2026
</h3>

<p align="center">
<strong>Junkai Tong</strong> · <strong>Mingjia Li</strong> · <strong>Haoran Chen</strong> ·
<strong>Yaoyu Jiang</strong> · <strong>Hanjie Ge</strong> · <strong>Yixuan Wang</strong> ·
<strong>Hong Qian</strong><sup>†</sup>
</p>

<p align="center">
School of Computer Science and Technology, East China Normal University, Shanghai, China
</p>

<div align="center"> <sup>†</sup>Corresponding Author. </div>

<p align="center">
    <a href="TODO-paper-link"><img src="https://img.shields.io/badge/Paper-ICDM2026-red"></a>
    <a href="TODO-github-link"><img src="https://img.shields.io/badge/GitHub-Repo-blue"></a>
</p>

<hr>

<h2 align="center">📰 [News]</h2>

<h3 align="center">[2026.09] 🎉 Our paper has been accepted to <strong>IEEE ICDM 2026</strong>.</h3>

<hr>

## 🧩 Overview

**ProIQA** is a *process-aware* framework for fine-grained math item quality
assessment. It models item quality across three dimensions—**knowledge
concept**, **difficulty**, and **disciplinary competency**—by augmenting the
item stem with a structured **reasoning tree** that explicitly represents the
mathematical solving process.

Specifically, ProIQA (i) uses an LLM to construct hierarchical reasoning trees
from raw solutions via top-down decomposition and bottom-up rationale update,
(ii) encodes their topological dependencies and procedural semantics with a
Graph Neural Network (GIN), and (iii) fuses the resulting solving
representation with stem semantics through a **dual-view ("Stem + Solving")**
architecture.

<hr>

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) model weights (or an alternative embedding backbone)

### Setup

```bash
pip install -r requirements.txt
```

Core dependencies: `torch 2.4`, `transformers 4.51`, `dgl 2.4` (must match your
CUDA version), `flash-attn 2.8`.

<hr>

## 🗂️ Repository Layout

```text
ProIQA/
|-- scripts/                        # Training, ablation, baseline, and analysis scripts
|   |-- train_concept.py            # Concept assessment (--dataset XES-500 / XES-1600)
|   |-- train_difficulty_algebra.py # Difficulty assessment (Algebra, 5-level classification)
|   |-- train_difficulty_pairwise.py# Difficulty assessment (XES-1500, pairwise ranking)
|   |-- train_competency.py         # Competency assessment (--dataset TIMSS19 / TIMSS23)
|   |-- ablation/                   # Ablation study (12 independent scripts)
|   |   |-- {concept,algebra,pairwise}_no_stream.py    # w/o Stream (linear solution text)
|   |   |-- {concept,algebra,pairwise}_no_process.py   # w/o Process (stem only)
|   |   |-- {concept,algebra,pairwise}_no_graph.py     # w/o Graph (whole-tree text)
|   |   `-- {concept,algebra,pairwise}_no_gnn.py       # w/o GNN (node pooling)
|   |-- baseline/                   # Supervised and zero-shot LLM baselines
|   |   |-- sft.py                  # Supervised fine-tuning (SFT)
|   |   `-- zero_shot_{concept,difficulty,pairwise}.py  # Zero-shot LLM baselines
|   |-- analysis/                   # Error-analysis and auxiliary scripts
|   |-- preprocess_concepts.py      # Extract knowledge concepts from reasoning trees
|   `-- preprocess_similarity.py    # Decomposition-example similarity preprocessing
|-- data/                           # Offline pre-built reasoning-tree data
|   |-- XES-500/   XES-1600/        # Concept assessment
|   |-- Algebra/   XES-1500/        # Difficulty assessment
|   `-- TIMSS19/   TIMSS23/         # Competency assessment
|-- prompts/                        # Solution/concept-extraction prompt templates
|-- PROMPTS.md                      # Reasoning-tree construction & verification prompts
|-- requirements.txt
|-- LICENSE
`-- README.md
```

<hr>

## 🚀 Usage

All training scripts are parameterized via `argparse`. Common arguments:
`--model_path` (pretrained embedding model path) and `--data_dir` (data root,
which defaults correctly relative to each script's location).

```bash
# Concept assessment (XES-1600, or --dataset XES-500)
python scripts/train_concept.py --model_path /path/to/Qwen3-Embedding-8B --dataset XES-1600

# Difficulty assessment (Algebra, 5-level classification)
python scripts/train_difficulty_algebra.py --model_path /path/to/Qwen3-Embedding-8B

# Difficulty assessment (XES-1500, pairwise ranking)
python scripts/train_difficulty_pairwise.py --model_path /path/to/Qwen3-Embedding-8B

# Competency assessment (default TIMSS23, --dataset TIMSS19 to switch)
python scripts/train_competency.py --model_path /path/to/Qwen3-Embedding-8B --dataset TIMSS23
```

Ablation scripts are under `scripts/ablation/` and follow the same interface.
Hyperparameters (learning rate, epochs, GNN layers, batch size, etc.) are at
the top of each script's `__main__` block.

Baseline scripts select their model via the `MODEL_PATH` environment variable
(`zero_shot_*.py` use a causal LM; `sft.py` uses an embedding model):

```bash
MODEL_PATH=/path/to/Qwen3-8B python scripts/baseline/zero_shot_concept.py
```

<hr>

## 📦 Data

Data are offline pre-built JSON files (stem, reasoning tree, and labels / IRT
parameters). Each dataset may split the **input** (stem + reasoning tree) and
the **labels** (concept / difficulty / competency) into separate files:

| Dataset | File | Content |
|---|---|---|
| XES-500 / XES-1600 (Concept) | `concept_items.json` | stem + concept labels + reasoning tree (input & labels combined) |
| XES-500 / XES-1600 (Concept) | `All_concepts.json` | full concept set (for the zero-shot baseline) |
| Algebra (Difficulty) | `difficulty_items.json` | stem + reasoning tree (GNN input) |
| Algebra (Difficulty) | `items.json` | difficulty level labels 1–5 (training target) |
| XES-1500 (Difficulty) | `difficulty_items.json` | stem + reasoning tree (GNN input) |
| XES-1500 (Difficulty) | `irt_parameters.json` | 2PL-IRT difficulty / discrimination parameters (training target) |
| TIMSS19 / TIMSS23 (Competency) | `items.json` | stem (solution text) + competency labels |

<hr>

## 💬 Prompts

The reasoning-tree construction and verification prompts are documented in
[PROMPTS.md](PROMPTS.md). Solution and concept-extraction prompt templates are
in `prompts/`.

<hr>

## 📊 Performance

Main results of ProIQA against LLM-based zero-shot evaluators and supervised
baselines (paper Table IV). **Bold** = best, *italic* = second-best, "—" = not
applicable.

*Main performance comparison among ProIQA, LLM-based zero-shot evaluators, and supervised baselines across concept, difficulty, and competency assessment tasks. Metrics are grouped by task: **Concept** (XES-500 / XES-1600 F1-Score), **Difficulty** (Algebra Level-ACC / Level-WACC; XES-1500 Pair-ACC), and **Competency** (TIMSS19 / TIMSS23 ACC). **Bold** indicates the best result, *italic* indicates the second-best, and "—" denotes not applicable.*

| Type | Method | XES-500 (F1) | XES-1600 (F1) | Level-ACC | Level-WACC | Pair-ACC | TIMSS19 (ACC) | TIMSS23 (ACC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Closed source | GPT-4o | 0.44±0.04 | 0.28±0.02 | 0.27±0.02 | 0.54±0.01 | 0.29±0.02 | 0.41±0.03 | 0.40±0.04 |
| Closed source | Qwen3-max | 0.51±0.04 | 0.36±0.02 | 0.30±0.01 | 0.60±0.01 | 0.30±0.02 | 0.46±0.04 | 0.32±0.05 |
| Closed source | Gemini-2.5-pro | 0.50±0.02 | 0.32±0.02 | 0.23±0.01 | 0.46±0.02 | 0.26±0.01 | 0.55±0.05 | 0.50±0.06 |
| Open source | DeepSeek-R1 | 0.53±0.02 | 0.35±0.01 | 0.24±0.01 | 0.47±0.02 | 0.26±0.02 | 0.36±0.05 | 0.35±0.05 |
| Open source | Qwen3-235B | 0.48±0.04 | 0.31±0.02 | 0.24±0.02 | 0.50±0.01 | 0.26±0.01 | 0.45±0.05 | 0.45±0.04 |
| Open source | Llama-3.3-70B | 0.41±0.03 | 0.26±0.01 | 0.31±0.02 | 0.61±0.01 | 0.34±0.02 | 0.37±0.05 | 0.33±0.05 |
| Open source | Qwen3-8B | 0.44±0.02 | 0.24±0.01 | 0.33±0.02 | 0.62±0.01 | 0.32±0.01 | 0.39±0.02 | 0.35±0.08 |
| Open source | Llama-3.1-8B | 0.23±0.02 | 0.09±0.00 | 0.20±0.02 | 0.44±0.02 | 0.28±0.01 | 0.37±0.07 | 0.37±0.05 |
| Open source | Mistral-7B-Instruct-v0.3 | 0.16±0.02 | 0.09±0.01 | 0.31±0.03 | 0.59±0.02 | 0.33±0.02 | 0.31±0.06 | 0.34±0.06 |
| Supervised | T-IRT | *0.70±0.04* | 0.65±0.01 | *0.50±0.03* | *0.73±0.02* | *0.45±0.01* | — | — |
| Supervised | R2DE | 0.41±0.04 | 0.43±0.01 | 0.42±0.02 | 0.67±0.02 | 0.37±0.02 | — | — |
| Supervised | Qwen3-8B (SFT) | 0.66±0.03 | 0.53±0.02 | 0.35±0.03 | 0.63±0.01 | 0.34±0.02 | 0.49±0.03 | 0.43±0.07 |
| Supervised | Llama-3.1-8B-Instruct (SFT) | 0.63±0.01 | 0.57±0.02 | 0.38±0.05 | 0.69±0.02 | 0.33±0.02 | 0.49±0.04 | 0.51±0.11 |
| Supervised | Mistral-7B-Instruct-v0.3 (SFT) | *0.70±0.04* | *0.70±0.02* | 0.39±0.06 | 0.67±0.07 | 0.33±0.02 | *0.61±0.04* | *0.58±0.08* |
| **Ours** | **ProIQA** | **0.75±0.03** | **0.75±0.02** | **0.57±0.02** | **0.81±0.01** | **0.49±0.03** | **0.73±0.04** | **0.71±0.09** |

<hr>

## 💭 Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{tong2026proiqa,
  title={ProIQA: A Process-Based Framework for Fine-Grained Math Item Quality Assessment},
  author={Tong, Junkai and Li, Mingjia and Chen, Haoran and Jiang, Yaoyu and Ge, Hanjie and Wang, Yixuan and Qian, Hong},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026},
  address={Shenyang, China}
}
```

<hr>

## ⚠️ Known Notes

1. **Tree-construction code is not included** here: the repository provides
   offline pre-built reasoning trees (`data/`) and the corresponding prompts
   (`PROMPTS.md`). The tree-construction script lives in a separate private
   environment.
2. **T-IRT and R2DE baselines are not included**; their results are reported
   in the paper (Table IV).
3. **SFT baseline data** (`scripts/baseline/sft.py`) is not included; prepare
   the raw training data before running it.
4. **Default `--model_path` / `MODEL_PATH`** point to the original training
   environment; specify your local model path.
5. **IRT calibration code is not included**: `data/XES-1500/irt_parameters.json`
   provides the pre-computed 2PL-IRT difficulty/discrimination parameters (see
   Section V of the paper). The script that fits the 2PL-IRT model to student
   response logs lives in a separate environment.
