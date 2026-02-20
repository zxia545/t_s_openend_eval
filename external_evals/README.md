# Local End-to-End Evaluation Framework

A unified framework to run **LiveBench**, **TruthfulQA**, **IFEval**, and **AlpacaEval2** on locally hosted models (e.g., using vLLM).

**📖 详细文档**: 请参阅 [EVALUATION_FRAMEWORK.md](./EVALUATION_FRAMEWORK.md) 了解每个评估的工作流程和数据流。

---

## 快速开始

### 1. 安装环境

```bash
conda create -n learnarena python=3.10
conda activate learnarena
pip install -r external_evals/tools/requirements.txt
```

### 2. 运行评估

```bash
# 方式 A: 评估单个 HuggingFace 模型
./external_evals/tools/run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct
./external_evals/tools/run_batch_evaluate.sh Qwen2.5-0.5B-Instruct

# 方式 B: 评估 SFT 训练产生的多个 checkpoint
./external_evals/tools/run_batch_generate.sh /path/to/checkpoints
./external_evals/tools/run_batch_evaluate.sh /path/to/checkpoints

# 方式 C: 只评估特定的 benchmarks
./external_evals/tools/run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa
```

---

## 四种评估基准

| Benchmark | 目的 | 输入 | 生成输出 | 评估输出 |
|-----------|------|------|----------|----------|
| **IFEval** | 指令跟随 | 541 条复杂指令 | `candidate_outputs.jsonl` | `eval_results_*.jsonl` |
| **TruthfulQA** | 真实性问答 | 790 个问题 | `TruthfulQA_generated.csv` | `results.csv` |
| **AlpacaEval2** | LLM-as-Judge | 805 个指令 | `model_outputs.json` | `annotations.json` |
| **LiveBench** | 多领域评估 | 多领域问题 | `model_answers/` | `model_judgment/` |

---

## 脚本使用说明

### run_batch_generate.sh - 生成阶段

```bash
./run_batch_generate.sh <model_input> [options]
```

**Model Input 类型:**
- 文件夹路径: `/path/to/checkpoints` (遍历所有子目录)
- HuggingFace 模型: `Qwen/Qwen2.5-0.5B-Instruct`
- 本地模型路径: `/root/models/my-model`

**选项:**
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-p, --port` | vLLM 服务端口 | 8001 |
| `-g, --gpu-mem` | GPU 内存使用率 | 0.85 |
| `-t, --tp-size` | Tensor Parallel 大小 | 1 |
| `-b, --benchmarks` | 要运行的 benchmarks | ifeval,truthfulqa,alpacaeval,livebench |
| `--skip-existing` | 跳过已有结果 | false |

**示例:**
```bash
# 评估 checkpoints 文件夹中的所有模型
./run_batch_generate.sh /root/checkpoints

# 评估单个 HuggingFace 模型
./run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct

# 只运行特定 benchmarks
./run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa

# 自定义端口和 GPU
./run_batch_generate.sh /root/models/my-model --port 8002 --gpu-mem 0.9
```

### run_batch_evaluate.sh - 评估阶段

```bash
./run_batch_evaluate.sh <model_input> [options]
```

**选项:**
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-b, --benchmarks` | 要评估的 benchmarks | ifeval,truthfulqa,alpacaeval,livebench |
| `--judge-api-base` | Judge API 地址 | OpenRouter |
| `--judge-api-key` | Judge API 密钥 | - |
| `--judge-model` | Judge 模型名称 | stepfun/step-3.5-flash:free |
| `--skip-existing` | 跳过已有评估结果 | false |

**示例:**
```bash
# 评估 checkpoints 文件夹中的所有结果
./run_batch_evaluate.sh /root/checkpoints

# 评估单个模型的结果
./run_batch_evaluate.sh Qwen2.5-0.5B-Instruct

# 使用本地 Judge 模型
./run_batch_evaluate.sh /root/checkpoints \
  --judge-api-base http://localhost:8000/v1 \
  --judge-model Qwen/Qwen2.5-72B-Instruct

# 只评估特定 benchmarks
./run_batch_evaluate.sh Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa
```

---

## 结果位置

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

## 典型工作流程

### 场景 1: 评估 SFT 训练的多个 checkpoint

```bash
# 假设你的 checkpoint 结构如下:
# /root/sft_checkpoints/
#   ├── step-100/
#   ├── step-200/
#   └── step-300/

# Step 1: 生成所有 checkpoint 的回答
./external_evals/tools/run_batch_generate.sh /root/sft_checkpoints

# Step 2: 评估所有回答
./external_evals/tools/run_batch_evaluate.sh /root/sft_checkpoints
```

### 场景 2: 评估原始预训练模型

```bash
# 直接使用 HuggingFace 模型名称
./external_evals/tools/run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct
./external_evals/tools/run_batch_evaluate.sh Qwen2.5-0.5B-Instruct
```

### 场景 3: 快速测试 (只跑 IFEval)

```bash
./external_evals/tools/run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct --benchmarks ifeval
./external_evals/tools/run_batch_evaluate.sh Qwen2.5-0.5B-Instruct --benchmarks ifeval
```

---

## 配置文件模式 (高级用法)

如果需要更细粒度的控制，可以创建配置文件:

```yaml
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
```

```bash
# 使用配置文件
python external_evals/tools/run_evals.py config.yaml --phase all
```
