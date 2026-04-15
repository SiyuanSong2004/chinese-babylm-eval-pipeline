# 2026 Chinese BabyLM Challenge — Evaluation Pipeline

<details open>
<summary><b>English</b></summary>

This repository contains the evaluation pipeline for the **2026 Chinese BabyLM Challenge**, adapted from the original BabyLM 2025 English evaluation pipeline. The challenge consists of three tracks:

| Track | Status | Description |
|---|---|---|
| **NLU Track** | Available | Minimal pairs (ZhoBLiMP) + fine-tuning (CLUE) |
| **Hanzi Track** | Available | Character structure and phonology minimal pairs |
| **Cog Track** | Available | Brain-aligned evaluation with fMRI data |

---
## Training data

- Option 1: you can use the 100M-word training data prepared by us. 

Download from huggingface: https://huggingface.co/datasets/chinese-babylm-org/babylm-zho-100M

100M words is calculated using jieba. 

- Option 2: BYO (build your own) training data.
As long as its size is 100M words, as tokenized by jieba. 


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
bash eval_finetuning.sh <model_path> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
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
bash eval_finetuning.sh <model_path> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
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

</details>
