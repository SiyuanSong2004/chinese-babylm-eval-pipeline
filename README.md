# 2026 Chinese BabyLM Challenge — Evaluation Pipeline

<details open>
<summary><b>Baselines</b></summary>
  
| Model | zhoblimp | hanzi_struc | hanzi_pinyin | word_fmri | fmri | afqmc | ocnli | tnews | cluewsc20 | mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-0.6B | 77.62 | 59.85 | 49.80 | 55.00 | 10.40 | 71.57 | 75.08 | 58.10 | 71.71 | 58.79 |
| Zh-Pythia-14M-v0.1-checkpoint-3000 | 57.10 | 52.00 | 11.50 | 55.70 | 7.80 | 69.05 | 57.66 | 51.62 | 63.49 | 47.32 |
| Zh-Pythia-70M-v0.1-checkpoint-3000 | 65.88 | 52.35 | 15.70 | 56.00 | 8.70 | 69.00 | 60.47 | 52.53 | 63.49 | 49.35 |
| Zh-Pythia-160M-v0.1-checkpoint-3000 | 70.98 | 52.40 | 18.50 | 56.00 | 8.90 | 68.98 | 60.58 | 52.91 | 63.49 | 50.30 |
| Zh-Pythia-410M-v0.1-checkpoint-3000 | 71.65 | 51.20 | 16.70 | 56.10 | 9.00 | 69.00 | 61.15 | 52.83 | 63.49 | 50.12 |
| Zh-Pythia-1.4B-v0.1-checkpoint-3000 | 75.67 | 53.80 | 17.10 | 56.10 | 9.00 | 69.00 | 61.97 | 53.84 | 63.82 | 51.14 |
| xlm-roberta-base | 84.00 | 57.90 | 37.90 | 55.60 | 10.60 | 73.03 | 73.93 | 56.05 | 63.49 | 56.94 |
| bert-base-chinese | 83.26 | 57.25 | 47.00 | 56.00 | 10.50 | 72.75 | 74.34 | 57.54 | 72.37 | 59.00 |
| chinese-bert-wwm-ext | 84.86 | 58.05 | 29.60 | 55.80 | 10.40 | 73.15 | 75.36 | 58.32 | 74.34 | 57.76 |
| roc-bert-base-zh | 53.89 | 56.80 | 35.50 | 55.60 | 9.90 | 69.00 | 37.39 | 57.57 | 63.49 | 48.79 |
| ChineseBERT-base | 51.69 | 56.25 | 46.80 | 55.60 | 4.70 | 69.00 | 37.39 | 27.78 | 63.49 | 45.86 |
| babylm-chinese-t5-14M-epoch3 | 57.40 | 51.70 | 28.00 | 55.50 | 8.10 | 69.00 | 42.00 | 37.30 | 63.80 | 45.87 |
| babylm-chinese-bert-14M-epoch3 | 59.19 | 54.35 | 35.70 | 55.90 | 9.00 | 69.00 | 52.92 | 52.18 | 62.83 | 50.12 |
| babylm-chinese-mamba-14M-epoch3 | 77.48 | 53.20 | 40.40 | 52.68 | 9.13 | 68.90 | 58.00 | 53.20 | 62.50 | 52.83 |
| babylm-chinese-pythia-14M-epoch3 | 73.29 | 52.05 | 30.10 | 53.70 | 8.30 | 69.14 | 56.31 | 52.65 | 61.84 | 50.82 |
| babylm-chinese-t5-14M-epoch20 | 63.58 | 52.70 | 39.20 | 53.30 | 7.99 | 69.00 | 47.63 | 50.95 | 65.46 |


</details>

<details open>
<summary><b>English</b></summary>

This repository contains the evaluation pipeline for the **2026 Chinese BabyLM Challenge**, adapted from the original BabyLM 2025 English evaluation pipeline. The challenge consists of three tracks:

| Track | Status | Description |
|---|---|---|
| **NLU Track** | Available | Minimal pairs (ZhoBLiMP) + fine-tuning (CLUE) |
| **Hanzi Track** | Available | Character structure and phonology minimal pairs |
| **Cog Track** | Available | Brain-aligned evaluation with fMRI data |

## Changelog

**[2026/04/19]**
1. Updated the ZhoBLiMP dataset (`chinese-babylm-org/zhoblimp`). Please use the latest version for all evaluations.
2. Added detailed rules for pre-training data — see the [Token Count](#token-count) section.
3. Removed unused English evaluation tasks from `eval_zero_shot.sh` and `eval_zero_shot_fast.sh`. Added `<backend>` as a required argument to `eval_finetuning.sh`.

---

## Quickstart: Integrated Pipeline

The easiest way to run the full evaluation is with `pipeline.py`, which handles data download, evaluation, and result reporting in two commands.

### 1. Download all evaluation data

```bash
python pipeline.py download
```

This downloads and prepares all datasets into `evaluation_data/` (ZhoBLiMP, Hanzi, CogBench fMRI, CLUE fine-tuning data).

### 2. Configure and run evaluation

Edit `config.yaml` to specify your models and tasks:

```yaml
models:
  - path: Qwen/Qwen3-0.6B   # HuggingFace ID or local path
    backend: causal

tasks:
  zero_shot:    [zhoblimp, hanzi_structure, hanzi_pinyin]
  cogbench:     [word_fmri, fmri]
  finetune:     [afqmc, ocnli, tnews, cluewsc2020]
```

Then run:

```bash
python pipeline.py eval --config config.yaml
```

Results are written to `results/` and a summary table is printed at the end:

```
=====================================================================
 Model          zhoblimp  hanzi_structure  hanzi_pinyin  word_fmri ...
=====================================================================
 Qwen3-0.6B        71.67             59.85          49.80       0.5495  ...
=====================================================================
```

**`pipeline.py eval` options:**

| Option | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to YAML config file |
| `--results_dir` | from config | Override results directory |

---

## Manual Evaluation

The individual shell scripts below remain available for running specific tracks separately.

## Setup

```bash
pip install -r requirements.txt
```

For gated HuggingFace datasets:

```bash
huggingface-cli login
```

---

## NLU Track

### Sentence Zero-Shot

Logit-based scoring of minimal pairs for Chinese linguistic phenomena (ZhoBLiMP).

```bash
bash eval_zero_shot.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix>
```

For intermediate checkpoints:

```bash
bash eval_zero_shot_fast.sh <model_path> <revision_name> <backend>

# All checkpoints at once:
bash eval_zero_shot_fast_all_revisions.sh <model_path> <backend> <track>
```

### Fine-Tuning

Sequence classification on Chinese CLUE tasks.

```bash
bash eval_finetuning.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
```

**Tasks:** AFQMC, OCNLI, TNEWS, CLUEWSC2020

---

## Hanzi Track

Logit-based scoring of minimal pairs targeting Chinese character knowledge.

| Task | Dataset | Phenomenon |
|---|---|---|
| `hanzi_structure` | `chinese-babylm-org/hanzi-structure` | Character structure (component relations) |
| `hanzi_pinyin` | `chinese-babylm-org/hanzi-pinyin` | Character phonology (pinyin similarity) |

```bash
bash eval_zero_shot.sh <model_path> <backend>
```

The hanzi tasks are included automatically when running `eval_zero_shot.sh`.

---

## Cog Track

Evaluates models against human brain recording data (fMRI). Uses ridge regression to fit model representations to neural signals, evaluated by Pearson/Spearman correlation.

### Data

Data is downloaded automatically by `python pipeline.py download`. To download manually:

```bash
# Downloads cogbench-fmri-0415.tar from zhiheng-qian/cogbench
# and extracts it to evaluation_data/cogbench-fmri-0415/
python -c "
import pathlib
from prepare_chinese_data import prepare_cogbench
prepare_cogbench(pathlib.Path('evaluation_data'))
"
```

The `train` and `dev` splits are available for development. The `test` split will be released later.

### Run

Sanity check (fast mode):

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --backend causal \
  --output_dir .
```

Full evaluation (default tasks: `word_fmri,fmri`):

```bash
bash eval_cogbench.sh \
  --model_path Qwen/Qwen3-0.6B \
  --output_dir .
```

Select specific tasks with `--task` (comma-separated):

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --task word_fmri \
  --output_dir .
```

**Available tasks:** `word_fmri`, `fmri` *(meg, eye_tracking: coming soon)*

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | HuggingFace model name or local path |
| `--backend` | `causal` | `causal`, `mlm`, `mntp`, `enc_dec_mask`, `enc_dec_prefix` |
| `--task` | `word_fmri,fmri` | Comma-separated task list |
| `--eval_dir` | `evaluation_data/cogbench-fmri-0415` | Path to data |
| `--output_dir` | current directory | Output directory |
| `--revision_name` | — | Model revision name |

### Model Loading

The pipeline loads your model via `transformers.AutoModel`. If the config is encoder-decoder, it falls back to `AutoModelForSeq2SeqLM`. For unsupported architectures, update `get_model_and_tokenizer` in `evaluation_pipeline/cogbench/utils/utils.py`.

By default, last hidden states are used as features. For encoder-decoder models, decoder hidden states are used.

---

## Results Structure

```
results/<model_name>/<revision_or_"main">/
  finetune/<task>/predictions.json + results.txt
  zero_shot/<backend>/<task>/<dataset>/predictions.json + best_temperature_report.txt
  cogbench/<task>/cogbench_<task>_<model_name>_report.json
```

## Collating Results

```bash
bash collate_preds.sh <model_name> <backend> <track>
```

## Evaluation Data

| Track | Task | Dataset | Source |
|---|---|---|---|
| NLU — Zero-shot | ZhoBLiMP | Minimal pairs (syntax, semantics, etc.) | `chinese-babylm-org/zhoblimp` |
| NLU — Fine-tuning | CLUE | AFQMC, OCNLI, TNEWS, CLUEWSC2020 | `clue` (HuggingFace) |
| Hanzi | hanzi-structure | Character component structure | `chinese-babylm-org/hanzi-structure` |
| Hanzi | hanzi-pinyin | Character phonology | `chinese-babylm-org/hanzi-pinyin` |
| Cog | CogBench | fMRI brain recordings | `zhiheng-qian/cogbench` |

## Token Count

The training data must contain **no more than 100M tokens**, where tokens are defined as **Jieba word segments** (version 0.42.1).

### Option 1 — Use the official pre-training dataset

The organizers provide a ready-to-use 100M-word dataset on HuggingFace Hub:

https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M

This dataset is derived from the original `BabyLM-community/babylm-zho` by removing a portion of WenetSpeech entries. It contains approximately 102M words as counted by jieba.

Load it directly via the `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("chinese-babylm-org/babylm-zho-100M")
```

The token count for this dataset has already been verified by the organizers using Jieba 0.42.1.

### Option 2 — BYO (build your own) training data

As long as its size is 100M words, as tokenized by jieba. If you choose a custom dataset, you must verify the token count yourself before submission. Count tokens with Jieba 0.42.1 as follows:

```python
import jieba  # version 0.42.1

total_tokens = 0
for text in your_texts:          # iterate over all training samples
    total_tokens += len(list(jieba.cut(text)))

print(f"Total Jieba tokens: {total_tokens:,}")
assert total_tokens <= 100_000_000, "Dataset exceeds 100M token limit"
```

**Rules:**
- Token count is always measured with `jieba.cut()` (default mode, no `HMM=False` or other flags).
- The Jieba version must be **0.42.1** (`pip install jieba==0.42.1`).
- Include all training splits in the count; evaluation/test splits are excluded.
- Report the exact token count in your system description.

</details>

---

<details>
<summary><b>中文</b></summary>

本仓库为 **2026 中文 BabyLM 挑战赛** 评测流水线，基于原版英文 BabyLM 2025 评测代码改编。挑战赛共分三个赛道：

| 赛道 | 状态 | 说明 |
|---|---|---|
| **NLU 赛道** | 可用 | 最小对评测（ZhoBLiMP）+ 微调（CLUE） |
| **汉字赛道** | 可用 | 汉字结构与语音最小对评测 |
| **Cog 赛道** | 可用 | 基于 fMRI 数据的脑对齐评测 |

## 更新日志

**[2026/04/19]**
1. 更新了 ZhoBLiMP 数据集（`chinese-babylm-org/zhoblimp`），测试请以最新版本为准。
2. 更新了关于预训练数据规则的详细说明，见 [Token 计数](#token-计数) 部分。
3. 移除了 `eval_zero_shot.sh` 和 `eval_zero_shot_fast.sh` 中未使用的英文评测任务。`eval_finetuning.sh` 新增必填参数 `<backend>`。

---

## 快速开始：集成流水线

推荐使用 `pipeline.py` 完成数据下载、评测和结果汇总，只需两条命令。

### 1. 下载所有评测数据

```bash
python pipeline.py download
```

自动下载并准备所有数据集到 `evaluation_data/`（ZhoBLiMP、汉字数据集、CogBench fMRI、CLUE 微调数据）。

### 2. 配置并运行评测

编辑 `config.yaml` 指定模型和任务：

```yaml
models:
  - path: Qwen/Qwen3-0.6B   # HuggingFace ID 或本地路径
    backend: causal

tasks:
  zero_shot:    [zhoblimp, hanzi_structure, hanzi_pinyin]
  cogbench:     [word_fmri, fmri]
  finetune:     [afqmc, ocnli, tnews, cluewsc2020]
```

然后运行：

```bash
python pipeline.py eval --config config.yaml
```

结果写入 `results/`，并在最后打印汇总表：

```
=====================================================================
 Model          zhoblimp  hanzi_structure  hanzi_pinyin  word_fmri ...
=====================================================================
 Qwen3-0.6B        67.3             72.1          58.4       0.31  ...
=====================================================================
```

**`pipeline.py eval` 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `config.yaml` | YAML 配置文件路径 |
| `--results_dir` | 来自配置文件 | 覆盖结果目录 |

---

## 手动评测

以下各赛道的 shell 脚本仍可单独使用。

## 环境配置

```bash
pip install -r requirements.txt
```

如需访问受限 HuggingFace 数据集：

```bash
huggingface-cli login
```

---

## NLU 赛道

### 句子零样本

基于对数概率对中文最小对进行语言现象评测（ZhoBLiMP）。

```bash
bash eval_zero_shot.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix>
```

中间检查点评测：

```bash
bash eval_zero_shot_fast.sh <model_path> <revision_name> <backend>

# 批量评测所有检查点：
bash eval_zero_shot_fast_all_revisions.sh <model_path> <backend> <track>
```

### 微调

在中文 CLUE 任务上进行序列分类微调。

```bash
bash eval_finetuning.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
```

**任务：** AFQMC, OCNLI, TNEWS, CLUEWSC2020

---

## 汉字赛道

基于对数概率对汉字知识最小对进行评测。

| 任务 | 数据集 | 测试内容 |
|---|---|---|
| `hanzi_structure` | `chinese-babylm-org/hanzi-structure` | 汉字结构（部件关系） |
| `hanzi_pinyin` | `chinese-babylm-org/hanzi-pinyin` | 汉字语音（拼音相似性） |

```bash
bash eval_zero_shot.sh <model_path> <backend>
```

运行 `eval_zero_shot.sh` 时汉字任务会自动包含在内。

---

## Cog 赛道

将模型表征与人脑神经记录数据（fMRI）对齐，使用岭回归拟合，以 Pearson/Spearman 相关系数评测。

### 数据

运行 `python pipeline.py download` 可自动下载。手动下载方式：

```bash
# 从 zhiheng-qian/cogbench 下载 cogbench-fmri-0415.tar
# 并解压至 evaluation_data/cogbench-fmri-0415/
python -c "
import pathlib
from prepare_chinese_data import prepare_cogbench
prepare_cogbench(pathlib.Path('evaluation_data'))
"
```

`train` 和 `dev` 分片可用于开发，`test` 分片将于后续发布。

### 运行

快速验证：

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --backend causal \
  --output_dir .
```

完整评测（默认任务：`word_fmri,fmri`）：

```bash
bash eval_cogbench.sh \
  --model_path Qwen/Qwen3-0.6B \
  --output_dir .
```

通过 `--task` 指定任务（逗号分隔）：

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --task word_fmri \
  --output_dir .
```

**可用任务：** `word_fmri`, `fmri` *（meg, eye_tracking 即将支持）*

**完整参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | 必填 | HuggingFace 模型名或本地路径 |
| `--backend` | `causal` | `causal`, `mlm`, `mntp`, `enc_dec_mask`, `enc_dec_prefix` |
| `--task` | `word_fmri,fmri` | 逗号分隔的任务列表 |
| `--eval_dir` | `evaluation_data/cogbench-fmri-0415` | 数据路径 |
| `--output_dir` | 当前目录 | 输出目录 |
| `--revision_name` | — | 模型版本名 |

### 模型加载

流水线通过 `transformers.AutoModel` 加载模型。如配置为编码器-解码器结构，则回退至 `AutoModelForSeq2SeqLM`。对于不支持的模型结构，请修改 `evaluation_pipeline/cogbench/utils/utils.py` 中的 `get_model_and_tokenizer`。

默认使用最后一层隐藏状态作为特征；编码器-解码器模型默认使用解码器隐藏状态。

---

## 结果目录结构

```
results/<model_name>/<revision_or_"main">/
  finetune/<task>/predictions.json + results.txt
  zero_shot/<backend>/<task>/<dataset>/predictions.json + best_temperature_report.txt
  cogbench/<task>/cogbench_<task>_<model_name>_report.json
```

## 汇总结果

```bash
bash collate_preds.sh <model_name> <backend> <track>
```

## 评测数据

| 赛道 | 任务 | 数据集 | 来源 |
|---|---|---|---|
| NLU — 零样本 | ZhoBLiMP | 最小对（句法、语义等） | `chinese-babylm-org/zhoblimp` |
| NLU — 微调 | CLUE | AFQMC, OCNLI, TNEWS, CLUEWSC2020 | `clue`（HuggingFace） |
| 汉字 | hanzi-structure | 汉字部件结构 | `chinese-babylm-org/hanzi-structure` |
| 汉字 | hanzi-pinyin | 汉字语音 | `chinese-babylm-org/hanzi-pinyin` |
| Cog | CogBench | fMRI 神经记录 | `zhiheng-qian/cogbench` |

## Token 计数

训练数据的 **总 Token 数不得超过 1 亿（100M）**，Token 以 **Jieba 分词**（版本 0.42.1）为计量标准。

### 方案 1 — 使用官方预训练数据集

主办方在 HuggingFace Hub 提供了经过验证的 100M 词数据集：

https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M

该数据集基于原始数据集 `BabyLM-community/babylm-zho` 移除部分 WenetSpeech 条目后得到，以 Jieba 统计约含 102M 词。

通过 `datasets` 库直接加载：

```python
from datasets import load_dataset
ds = load_dataset("chinese-babylm-org/babylm-zho-100M")
```

该数据集的 Token 数已由主办方使用 Jieba 0.42.1 完成核验，参赛者可直接使用，无需自行统计。

### 方案 2 — 使用自选数据集（BYO）

只要词数不超过 100M（以 Jieba 分词为准）即可。若选择自定义数据集，须在提交前自行核验词数总量。请按以下方式使用 Jieba 0.42.1 统计：

```python
import jieba  # 版本 0.42.1

total_tokens = 0
for text in your_texts:          # 遍历所有训练样本的文本
    total_tokens += len(list(jieba.cut(text)))

print(f"Jieba 总词数：{total_tokens:,}")
assert total_tokens <= 100_000_000, "数据集超出 100M Token 上限"
```

**规则说明：**
- Token 计数统一使用 `jieba.cut()`（默认模式，不得添加 `HMM=False` 等参数）。
- Jieba 版本须为 **0.42.1**（安装命令：`pip install jieba==0.42.1`）。
- 计数范围覆盖全部训练分片；评测集和测试集不计入。
- 请在系统说明（system description）中报告确切的 Token 总数。

</details>
