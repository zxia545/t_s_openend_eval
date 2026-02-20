# Evaluation Framework 完整指南

本文档详细说明了四个评估基准测试的工作流程、数据流和输出位置。

---

## 目录

1. [概述](#概述)
2. [四种评估基准详解](#四种评估基准详解)
   - [IFEval](#1-ifeval---指令跟随评估)
   - [TruthfulQA](#2-truthfulqa---真实性问答)
   - [AlpacaEval2](#3-alpacaeval2---指令遵循评估)
   - [LiveBench](#4-livebench---实时基准测试)
3. [数据流总览](#数据流总览)
4. [使用方法](#使用方法)
5. [结果解读](#结果解读)

---

## 概述

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Evaluation Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   vLLM      │    │   Candidate │    │  Results/   │                      │
│  │   Server    │───▶│   Model     │───▶│  <model_id>/│                      │
│  │  (Port 8001)│    │   Answers   │    │              │                      │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                      │
│                                               │                              │
│                    GENERATION PHASE           │                              │
│  ────────────────────────────────────────────┼───────────────────────────── │
│                                               │                              │
│                                               ▼                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │  OpenRouter │    │   Judge     │    │   Final     │                      │
│  │   API       │───▶│   Model     │───▶│   Metrics   │                      │
│  │             │    │             │    │              │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│                                                                              │
│                    EVALUATION PHASE                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四种评估基准详解

### 1. IFEval - 指令跟随评估

**目的**: 测试模型是否能够精确遵循复杂指令（如格式约束、长度要求等）

#### 数据流

```
输入数据:
external_evals/IFEval/instruction_following_eval/data/input_data.jsonl
├── 541 条测试指令
├── 每条包含: prompt, instruction_id_list, kwargs
└── 指令类型: punctuation, detectable_format, length_constraints, etc.

生成阶段 (run_evals.py):
1. 读取 input_data.jsonl
2. 调用候选模型 API 生成回答
3. 输出: candidate_outputs.jsonl
   └── 位置: results/<model_id>/ifeval/candidate_outputs.jsonl

评估阶段 (evaluation_main.py):
1. 读取 candidate_outputs.jsonl
2. 使用规则检查器验证每条指令
3. 输出:
   ├── eval_results_strict.jsonl  (严格匹配)
   └── eval_results_loose.jsonl   (宽松匹配)
   └── 位置: results/<model_id>/ifeval/
```

#### 输入文件格式
```json
{
  "key": 1000,
  "prompt": "Write a 300+ word summary...",
  "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
  "kwargs": [{}, {"relation": "at least", "num_words": 300}]
}
```

#### 生成文件格式 (candidate_outputs.jsonl)
```json
{"prompt": "...", "response": "..."}
```

#### 评估文件格式 (eval_results_*.jsonl)
```json
{
  "follow_all_instructions": false,
  "follow_instruction_list": [false, true, false],
  "instruction_id_list": ["punctuation:no_comma", ...],
  "prompt": "...",
  "response": "..."
}
```

#### 指标计算
```python
strict_accuracy = sum(1 for r in results if r['follow_all_instructions']) / total
loose_accuracy = sum(1 for r in results if any(r['follow_instruction_list'])) / total
```

---

### 2. TruthfulQA - 真实性问答

**目的**: 测试模型是否会生成常见的误解或虚假信息

#### 数据流

```
输入数据:
external_evals/TruthfulQA/TruthfulQA.csv
├── 790 个问题
├── 列: Type, Category, Question, Best Answer, Correct Answers, Incorrect Answers
└── 涵盖: Misconceptions, Politics, Fiction, etc.

生成阶段 (run_evals.py):
1. 读取 TruthfulQA.csv
2. 格式化 prompt: "Q: {question}\nA:"
3. 调用候选模型 API 生成回答
4. 输出: TruthfulQA_generated.csv
   └── 位置: results/<model_id>/truthfulqa/TruthfulQA_generated.csv

评估阶段 (truthfulqa/evaluate.py):
1. 读取 TruthfulQA_generated.csv
2. 计算与正确/错误答案的相似度 (BLEU, ROUGE, BLEURT)
3. 输出: results.csv
   └── 位置: results/<model_id>/truthfulqa/results.csv
```

#### 输入文件格式 (TruthfulQA.csv)
```csv
Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source
Adversarial,Misconceptions,What happens if you eat watermelon seeds?,...,The watermelon seeds pass through...
```

#### 生成文件格式 (TruthfulQA_generated.csv)
```csv
(原始列) + <model_id> (新增列，包含模型回答)
```

#### 评估文件格式 (results.csv)
```csv
(原始列) + <model_id>, <model_id> bleu max, <model_id> rouge1 max, 
          <model_id> bleu acc, <model_id> rouge1 acc, ...
```

#### 指标说明
- **bleu/rouge max**: 与最佳答案的相似度
- **bleu/rouge acc**: 回答是否更接近正确答案而非错误答案
- **Truthful Score**: acc 指标的平均值

---

### 3. AlpacaEval2 - 指令遵循评估

**目的**: 使用 LLM-as-Judge 评估模型回答质量

#### 数据流

```
输入数据:
HuggingFace Dataset: tatsu-lab/alpaca_eval
├── 805 个指令
├── 每条包含: instruction, output (baseline), dataset
└── 来源: AlpacaFarm evaluation set

生成阶段 (run_evals.py):
1. 从 HuggingFace 加载数据集
2. 调用候选模型 API 生成回答
3. 输出: model_outputs.json
   └── 位置: results/<model_id>/alpacaeval2/model_outputs.json

评估阶段 (alpaca_eval evaluate):
1. 读取 model_outputs.json
2. Judge 模型比较候选回答 vs baseline 回答
3. 使用 JSON Function Calling (非 logprobs)
4. 输出:
   ├── annotations.json          (每条对比结果)
   ├── leaderboard.csv           (汇总指标)
   └── 位置: alpaca_eval/results/<model_id>/
```

#### 生成文件格式 (model_outputs.json)
```json
[
  {
    "instruction": "Write a poem about AI.",
    "output": "In circuits deep...",
    "generator": "Qwen2.5-0.5B-Instruct",
    "dataset": "alpaca_eval"
  },
  ...
]
```

#### 评估配置 (custom_vllm_judge/configs.yaml)
```yaml
custom_vllm_judge:
  prompt_template: "alpaca_eval_gpt4_turbo_fn/alpaca_eval_fn.txt"
  fn_completions: "openai_completions"
  completions_kwargs:
    model_name: "stepfun/step-3.5-flash:free"
    max_tokens: 100
    temperature: 0.0
    tool_choice:
      type: function
      function:
        name: make_partial_leaderboard
```

#### 指标说明
- **Win Rate**: 候选回答优于 baseline 的比例
- **Standard Error**: 置信区间

---

### 4. LiveBench - 实时基准测试

**目的**: 多领域、多任务的综合评估

#### 数据流

```
输入数据:
HuggingFace Dataset: livebench/livebench
├── 多个领域: math, coding, reasoning, etc.
├── 从 HuggingFace 动态加载
└── Release: 2024-11-25

生成阶段 (gen_api_answer.py):
1. 从 HuggingFace 加载问题
2. 调用候选模型 API 生成回答
3. 输出:
   ├── model_answers/<model_id>/<category>.jsonl
   └── 位置: livebench/livebench/model_answers/

评估阶段 (gen_ground_truth_judgment.py):
1. 读取模型回答
2. Judge 模型评分 (1-10 分)
3. 输出:
   ├── model_judgment/<model_id>/<category>.jsonl
   └── 位置: livebench/livebench/model_judgment/
```

#### 生成文件格式 (model_answers/*.jsonl)
```json
{
  "question_id": "...",
  "model": "Qwen2.5-0.5B-Instruct",
  "answer": "...",
  "category": "math"
}
```

#### 评估文件格式 (model_judgment/*.jsonl)
```json
{
  "question_id": "...",
  "model": "Qwen2.5-0.5B-Instruct",
  "judgment": {
    "score": 8,
    "reasoning": "..."
  }
}
```

---

## 数据流总览

### 文件位置映射表

| 阶段 | Benchmark | 输入位置 | 输出位置 |
|------|-----------|----------|----------|
| **生成** | IFEval | `IFEval/.../input_data.jsonl` | `results/<model>/ifeval/candidate_outputs.jsonl` |
| **生成** | TruthfulQA | `TruthfulQA/TruthfulQA.csv` | `results/<model>/truthfulqa/TruthfulQA_generated.csv` |
| **生成** | AlpacaEval2 | `HuggingFace: tatsu-lab/alpaca_eval` | `results/<model>/alpacaeval2/model_outputs.json` |
| **生成** | LiveBench | `HuggingFace: livebench/livebench` | `livebench/livebench/model_answers/<model>/` |
| **评估** | IFEval | `results/<model>/ifeval/candidate_outputs.jsonl` | `results/<model>/ifeval/eval_results_*.jsonl` |
| **评估** | TruthfulQA | `results/<model>/truthfulqa/TruthfulQA_generated.csv` | `results/<model>/truthfulqa/results.csv` |
| **评估** | AlpacaEval2 | `results/<model>/alpacaeval2/model_outputs.json` | `alpaca_eval/results/<model>/` |
| **评估** | LiveBench | `livebench/livebench/model_answers/<model>/` | `livebench/livebench/model_judgment/<model>/` |

### 结果目录结构

```
external_evals/results/
└── <model_id>/
    ├── ifeval/
    │   ├── candidate_outputs.jsonl        # 生成结果
    │   ├── eval_results_strict.jsonl      # 评估结果 (严格)
    │   └── eval_results_loose.jsonl       # 评估结果 (宽松)
    │
    ├── truthfulqa/
    │   ├── TruthfulQA_generated.csv       # 生成结果
    │   └── results.csv                    # 评估结果
    │
    └── alpacaeval2/
        └── model_outputs.json             # 生成结果

external_evals/alpaca_eval/results/
└── <model_id>/
    ├── annotations.json                   # 评估结果 (详细)
    └── leaderboard.csv                    # 评估结果 (汇总)

external_evals/livebench/livebench/
├── model_answers/
│   └── <model_id>/
│       └── <category>.jsonl               # 生成结果
└── model_judgment/
    └── <model_id>/
        └── <category>.jsonl               # 评估结果
```

---

## 使用方法

### 方式 1: 灵活输入模式 (推荐)

#### 生成阶段

```bash
# 场景 1: 评估一个 checkpoints 文件夹 (SFT 训练产生的多个 checkpoint)
./run_batch_generate.sh /path/to/checkpoints

# 场景 2: 评估单个预训练模型 (直接输入模型名称)
./run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct

# 场景 3: 评估 HuggingFace 上的模型
./run_batch_generate.sh meta-llama/Llama-3-8b-chat-hf

# 场景 4: 评估本地模型路径
./run_batch_generate.sh /root/models/my-finetuned-model

# 可选参数
./run_batch_generate.sh <model_path> [port] [gpu_mem] [tp_size]
```

#### 评估阶段

```bash
# 场景 1: 评估 checkpoints 文件夹的结果
./run_batch_evaluate.sh /path/to/checkpoints

# 场景 2: 评估单个模型的结果 (使用模型 ID)
./run_batch_evaluate.sh Qwen2.5-0.5B-Instruct

# 场景 3: 指定 Judge 模型 (默认使用 OpenRouter)
./run_batch_evaluate.sh <model_id_or_path> --judge-model gpt-4

# 场景 4: 只评估特定 benchmarks
./run_batch_evaluate.sh <model_id_or_path> --benchmarks ifeval,truthfulqa
```

### 方式 2: 单次运行模式

```bash
# 创建配置文件
cat > config.yaml << EOF
models:
  - id: "my-model"
    api_name: "my-model"
    api_base: "http://localhost:8001/v1"
    api_key: "dummy"

judge:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "sk-or-v1-xxx"
  model_name: "stepfun/step-3.5-flash:free"

benchmarks:
  ifeval: true
  truthfulqa: true
  alpacaeval: true
  livebench: true
EOF

# 运行
python run_evals.py config.yaml --phase all
```

---

## 结果解读

### IFEval 结果

```python
# 从 eval_results_strict.jsonl 计算
strict_accuracy = 25.69%  # 完全遵循所有指令的比例
loose_accuracy = 28.28%   # 至少遵循一条指令的比例
```

### TruthfulQA 结果

```python
# 从 results.csv 计算
{
    "BLEU Accuracy": 41.39%,    # BLEU 相似度准确性
    "ROUGE-1 Accuracy": 46.20%, # ROUGE-1 相似度准确性
    "ROUGE-2 Accuracy": 31.77%, # ROUGE-2 相似度准确性
    "ROUGE-L Accuracy": 42.53%, # ROUGE-L 相似度准确性
    "Overall Truthful": 40.47%  # 平均准确性
}
```

### AlpacaEval2 结果

```python
# 从 leaderboard.csv 读取
{
    "Win Rate": 0.xxx,         # 胜率 (0-1)
    "Standard Error": 0.xxx,   # 标准误差
    "Avg Length": xxx          # 平均回答长度
}
```

### LiveBench 结果

```python
# 从 model_judgment/ 计算
{
    "overall_score": 5.23,     # 平均分数 (1-10)
    "math_score": 4.5,
    "coding_score": 6.1,
    ...
}
```

---

## 常见问题

### Q1: 为什么 AlpacaEval2 评估很慢?

A: 因为我们使用 OpenRouter 的免费 Judge 模型 (`stepfun/step-3.5-flash:free`)，有 50 RPM 速率限制。805 条数据需要约 16 分钟。

### Q2: 可以使用本地 Judge 模型吗?

A: 可以。修改 `run_batch_evaluate.sh` 中的 judge 配置:
```bash
--judge-api-base "http://localhost:8000/v1"
--judge-model "Qwen/Qwen2.5-72B-Instruct"
```

### Q3: 如何只运行某个 benchmark?

A: 在配置文件中设置:
```yaml
benchmarks:
  ifeval: true
  truthfulqa: false
  alpacaeval: false
  livebench: false
```

### Q4: 生成阶段可以跳过已完成的模型吗?

A: 可以。如果输出文件已存在，脚本会自动跳过。

---

## 依赖说明

```
# 必需
- Python 3.10+
- vLLM (用于本地模型推理)
- OpenAI API 兼容的 API 服务

# Python 包
- openai
- datasets (HuggingFace)
- yaml
- pandas
- evaluate, bleurt, t5 (TruthfulQA 评估)
- alpaca_eval
```
