#!/usr/bin/env bash

set -euo pipefail

DEFAULT_BENCHMARKS="ifeval,truthfulqa,alpacaeval,livebench"

show_help() {
    cat << EOF
Run Generate + Evaluate in one command

用法:
    $0 [model_input] [options]

参数:
    model_input            必需，待评测模型或checkpoint目录

选项:
    -b, --benchmarks LIST  benchmark 列表 (默认: $DEFAULT_BENCHMARKS)

    # Generate 选项
    -p, --port PORT
    -g, --gpu-mem UTIL
    -t, --tp-size SIZE
    --gpus LIST
    --num-workers N
    --stagger-seconds S
    --no-parallel

    # Evaluate Judge 选项
    --judge-api-base URL   默认建议: vllm
    --judge-api-key KEY
    --judge-model NAME_OR_PATH
    --judge-port PORT
    --judge-gpu-mem UTIL
    --judge-tp-size SIZE

    --skip-existing        两阶段都跳过已有结果
    --dry-run              两阶段都只打印计划
    -h, --help             显示帮助

EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/external_evals/tools"
GEN_SCRIPT="$TOOLS_DIR/run_batch_generate.sh"
EVAL_SCRIPT="$TOOLS_DIR/run_batch_evaluate.sh"

MODEL_INPUT=""
BENCHMARKS="$DEFAULT_BENCHMARKS"

GEN_ARGS=()
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        -p|--port|-g|--gpu-mem|-t|--tp-size|--gpus|--num-workers|--stagger-seconds)
            GEN_ARGS+=("$1" "$2")
            shift 2
            ;;
        --no-parallel)
            GEN_ARGS+=("$1")
            shift
            ;;
        --judge-api-base|--judge-api-key|--judge-model|--judge-port|--judge-gpu-mem|--judge-tp-size)
            EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        --skip-existing)
            GEN_ARGS+=("$1")
            EVAL_ARGS+=("$1")
            shift
            ;;
        --dry-run)
            GEN_ARGS+=("$1")
            EVAL_ARGS+=("$1")
            shift
            ;;
        -* )
            echo "错误: 未知选项 $1"
            show_help
            exit 1
            ;;
        *)
            MODEL_INPUT="$1"
            shift
            ;;
    esac
done

echo "============================================================"
echo "            LLM Evaluation - All Phases"
echo "============================================================"
echo ""

if [ -z "$MODEL_INPUT" ]; then
    echo "错误: 必须指定 model_input"
    show_help
    exit 1
fi

echo "配置:"
echo "  Model Input: $MODEL_INPUT"
echo "  Benchmarks:  $BENCHMARKS"
echo ""

echo "[PHASE 1/2] Generate"
"$GEN_SCRIPT" "$MODEL_INPUT" --benchmarks "$BENCHMARKS" "${GEN_ARGS[@]}"

echo ""
echo "[PHASE 2/2] Evaluate"
"$EVAL_SCRIPT" "$MODEL_INPUT" --benchmarks "$BENCHMARKS" "${EVAL_ARGS[@]}"

echo ""
echo "[DONE] 全流程完成"
