#!/usr/bin/env bash
#
# Flexible Batch Evaluation Script
# 
# 用法:
#   ./run_batch_evaluate.sh <model_input> [options]
#
# Model Input 类型:
#   1. 文件夹路径: /path/to/checkpoints  (遍历所有子目录的结果)
#   2. 模型 ID: Qwen2.5-0.5B-Instruct  (直接指定结果目录)
#
# 示例:
#   # 评估 checkpoints 文件夹中所有模型的结果
#   ./run_batch_evaluate.sh /root/checkpoints
#
#   # 评估单个模型的结果
#   ./run_batch_evaluate.sh Qwen2.5-0.5B-Instruct
#
#   # 使用本地 Judge 模型
#   ./run_batch_evaluate.sh /root/checkpoints --judge-api-base http://localhost:8000/v1
#
#   # 只评估特定 benchmarks
#   ./run_batch_evaluate.sh Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa
#

set -euo pipefail

# ==================== 默认配置 ====================
DEFAULT_BENCHMARKS="ifeval,truthfulqa,alpacaeval,livebench"
DEFAULT_JUDGE_API_BASE="https://openrouter.ai/api/v1"
DEFAULT_JUDGE_API_KEY="sk-or-v1-sf"
DEFAULT_JUDGE_MODEL="stepfun/step-3.5-flash:free"

# ==================== 帮助信息 ====================
show_help() {
    cat << EOF
Flexible Batch Evaluation Script for LLM Evaluation

用法:
    $0 <model_input> [options]

参数:
    model_input          模型输入 (必需)
                         - 文件夹: 遍历所有子目录的结果
                         - 模型 ID: 直接指定结果目录

选项:
    -b, --benchmarks LIST  要评估的 benchmarks (逗号分隔)
                           (默认: $DEFAULT_BENCHMARKS)
    --judge-api-base URL   Judge API 地址
                           (默认: OpenRouter)
    --judge-api-key KEY    Judge API 密钥
    --judge-model NAME     Judge 模型名称
                           (默认: $DEFAULT_JUDGE_MODEL)
    --skip-existing        跳过已有评估结果的模型
    -h, --help             显示此帮助信息

示例:
    # 评估 checkpoints 文件夹中的所有结果
    $0 /path/to/checkpoints

    # 评估单个模型的结果
    $0 Qwen2.5-0.5B-Instruct

    # 使用本地 Judge 模型
    $0 /path/to/checkpoints --judge-api-base http://localhost:8000/v1 --judge-model Qwen/Qwen2.5-72B-Instruct

    # 只评估特定 benchmarks
    $0 Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa

EOF
}

# ==================== 参数解析 ====================
MODEL_INPUT=""
BENCHMARKS=$DEFAULT_BENCHMARKS
JUDGE_API_BASE=$DEFAULT_JUDGE_API_BASE
JUDGE_API_KEY=$DEFAULT_JUDGE_API_KEY
JUDGE_MODEL=$DEFAULT_JUDGE_MODEL
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --judge-api-base)
            JUDGE_API_BASE="$2"
            shift 2
            ;;
        --judge-api-key)
            JUDGE_API_KEY="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "错误: 未知选项 $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$MODEL_INPUT" ]; then
                MODEL_INPUT="$1"
            else
                echo "错误: 只能指定一个 model_input"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_INPUT" ]; then
    echo "错误: 必须指定 model_input"
    show_help
    exit 1
fi

# ==================== 路径设置 ====================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/external_evals/tools"
RESULTS_DIR="$ROOT_DIR/external_evals/results"

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate learnarena 2>/dev/null || true

# ==================== 辅助函数 ====================

# 检查是否已有评估结果
check_existing_eval_results() {
    local model_id=$1
    local has_results=true
    
    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | tr -d ' ')
        case $bench in
            ifeval)
                [ -f "$RESULTS_DIR/$model_id/ifeval/eval_results_strict.jsonl" ] || has_results=false
                ;;
            truthfulqa)
                [ -f "$RESULTS_DIR/$model_id/truthfulqa/results.csv" ] || has_results=false
                ;;
            alpacaeval)
                [ -f "$ROOT_DIR/external_evals/alpaca_eval/results/$model_id/leaderboard.csv" ] || has_results=false
                ;;
            livebench)
                [ -d "$ROOT_DIR/external_evals/livebench/livebench/model_judgment/$model_id" ] || has_results=false
                ;;
        esac
    done
    
    $has_results
}

# 生成 YAML 配置
generate_config() {
    local model_id=$1
    local config_file=$2
    
    # 构建 benchmark 配置
    local bench_yaml=""
    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | tr -d ' ')
        bench_yaml="$bench_yaml  $bench: true\n"
    done
    
    cat > "$config_file" << EOF
models:
  - id: "${model_id}"
    api_name: "${model_id}"
    api_base: "http://localhost:8001/v1"
    api_key: "dummy"

judge:
  api_base: "${JUDGE_API_BASE}"
  api_key: "${JUDGE_API_KEY}"
  model_name: "${JUDGE_MODEL}"

benchmarks:
$(echo -e "$bench_yaml")
EOF
}

# 运行单个模型的评估
run_model_evaluate() {
    local model_id=$1
    
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVALUATION: $model_id"
    echo "=========================================================="
    
    # 检查是否跳过
    if [ "$SKIP_EXISTING" = true ] && check_existing_eval_results "$model_id"; then
        echo "[INFO] 评估结果已存在，跳过: $model_id"
        return 0
    fi
    
    # 检查生成结果是否存在
    local has_generate_results=false
    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | tr -d ' ')
        case $bench in
            ifeval)
                [ -f "$RESULTS_DIR/$model_id/ifeval/candidate_outputs.jsonl" ] && has_generate_results=true
                ;;
            truthfulqa)
                [ -f "$RESULTS_DIR/$model_id/truthfulqa/TruthfulQA_generated.csv" ] && has_generate_results=true
                ;;
            alpacaeval)
                [ -f "$RESULTS_DIR/$model_id/alpacaeval2/model_outputs.json" ] && has_generate_results=true
                ;;
        esac
    done
    
    if [ "$has_generate_results" = false ]; then
        echo "[WARN] 没有找到生成结果，跳过: $model_id"
        return 0
    fi
    
    # 生成配置
    local config_file="$TOOLS_DIR/config_eval_${model_id}.yaml"
    generate_config "$model_id" "$config_file"
    
    # 运行评估
    echo "[INFO] 开始评估 (Judge: $JUDGE_MODEL)..."
    if python "$TOOLS_DIR/run_evals.py" "$config_file" --phase evaluate; then
        echo "[INFO] ✓ 评估完成: $model_id"
    else
        echo "[ERROR] ✗ 评估失败: $model_id"
    fi
    
    # 清理
    rm -f "$config_file"
    
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: $model_id"
    echo "=========================================================="
}

# 获取模型 ID 列表
get_model_ids() {
    local input=$1
    
    if [ -d "$input" ]; then
        # 文件夹 - 检查是 checkpoints 目录还是 results 目录
        if [[ "$input" == *"results"* ]]; then
            # results 目录
            ls -1 "$input" 2>/dev/null
        else
            # checkpoints 目录 - 获取子目录名
            find "$input" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; 2>/dev/null
        fi
    else
        # 直接是模型 ID
        echo "$input"
    fi
}

# ==================== 主逻辑 ====================

echo "============================================================"
echo "            LLM Evaluation - Evaluation Phase"
echo "============================================================"
echo ""
echo "配置:"
echo "  Model Input:    $MODEL_INPUT"
echo "  Benchmarks:     $BENCHMARKS"
echo "  Judge API:      $JUDGE_API_BASE"
echo "  Judge Model:    $JUDGE_MODEL"
echo "  Skip Existing:  $SKIP_EXISTING"
echo ""

# 获取模型 ID 列表
MODEL_IDS=$(get_model_ids "$MODEL_INPUT")

if [ -z "$MODEL_IDS" ]; then
    echo "[ERROR] 没有找到任何模型"
    exit 1
fi

# 统计模型数量
MODEL_COUNT=$(echo "$MODEL_IDS" | wc -l)
echo "[INFO] 找到 $MODEL_COUNT 个模型"
echo ""

# 遍历评估
for model_id in $MODEL_IDS; do
    run_model_evaluate "$model_id"
done

echo ""
echo "============================================================"
echo "            所有评估任务完成!"
echo "============================================================"
echo ""
echo "结果位置:"
echo "  - IFEval:     $RESULTS_DIR/<model_id>/ifeval/eval_results_*.jsonl"
echo "  - TruthfulQA: $RESULTS_DIR/<model_id>/truthfulqa/results.csv"
echo "  - AlpacaEval: $ROOT_DIR/external_evals/alpaca_eval/results/<model_id>/"
echo "  - LiveBench:  $ROOT_DIR/external_evals/livebench/livebench/model_judgment/<model_id>/"
