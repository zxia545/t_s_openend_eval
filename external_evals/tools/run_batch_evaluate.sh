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
DEFAULT_JUDGE_API_BASE="vllm"
DEFAULT_JUDGE_API_KEY="dummy"
DEFAULT_JUDGE_MODEL="/home/aiscuser/zhengyu_blob_home/hugging_face_models/model--Qwen-Qwen2.5-72B-Instruct"
DEFAULT_JUDGE_PORT=8065
DEFAULT_JUDGE_GPU_MEM=0.85
DEFAULT_JUDGE_TP_SIZE=0

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
                           (默认: vllm)
    --judge-api-key KEY    Judge API 密钥
    --judge-model NAME     Judge 模型名称
                            (默认: $DEFAULT_JUDGE_MODEL)
    --judge-port PORT      本地 vLLM Judge 端口 (默认: $DEFAULT_JUDGE_PORT)
    --judge-gpu-mem UTIL   本地 vLLM Judge GPU 内存使用率 (默认: $DEFAULT_JUDGE_GPU_MEM)
    --judge-tp-size SIZE   本地 vLLM Judge TP 大小，0=自动=GPU数量
                           (默认: $DEFAULT_JUDGE_TP_SIZE)
    --dry-run              只打印计划，不实际执行评估
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
JUDGE_PORT=$DEFAULT_JUDGE_PORT
JUDGE_GPU_MEM=$DEFAULT_JUDGE_GPU_MEM
JUDGE_TP_SIZE=$DEFAULT_JUDGE_TP_SIZE
SKIP_EXISTING=false
DRY_RUN=false

LOCAL_JUDGE_MODE=false
LOCAL_JUDGE_VLLM_PID=""
LOCAL_JUDGE_MODEL_API_NAME=""

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
        --judge-port)
            JUDGE_PORT="$2"
            shift 2
            ;;
        --judge-gpu-mem)
            JUDGE_GPU_MEM="$2"
            shift 2
            ;;
        --judge-tp-size)
            JUDGE_TP_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

sanitize_for_filename() {
    echo "$1" | tr '/: ' '___'
}

split_csv_to_array() {
    local csv="$1"
    local -n out_arr="$2"
    out_arr=()
    if [ -z "$csv" ]; then
        return 0
    fi
    IFS=',' read -ra out_arr <<< "$csv"
    for i in "${!out_arr[@]}"; do
        out_arr[$i]="$(echo "${out_arr[$i]}" | tr -d ' ')"
    done
}

detect_gpu_ids() {
    local -n out_arr="$1"
    out_arr=()

    if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
        split_csv_to_array "$CUDA_VISIBLE_DEVICES" out_arr
        if [ ${#out_arr[@]} -gt 0 ] && [ -n "${out_arr[0]}" ]; then
            return 0
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t out_arr < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
        if [ ${#out_arr[@]} -gt 0 ]; then
            return 0
        fi
    fi

    out_arr=("0")
}

wait_for_server() {
    local port=$1
    local timeout=1000
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

should_use_local_vllm_judge() {
    [ "$JUDGE_API_BASE" = "vllm" ]
}

try_reuse_existing_local_judge() {
    local api_base="http://localhost:${JUDGE_PORT}/v1"
    local models_url="${api_base}/models"
    local response
    response=$(curl -s --max-time 3 "$models_url" 2>/dev/null || true)

    if [ -z "$response" ]; then
        return 1
    fi

    local detected_model
    detected_model=$(
        RESPONSE="$response" REQUESTED_MODEL="$JUDGE_MODEL" python - <<'PY'
import json
import os

response = os.environ.get("RESPONSE", "")
requested = os.environ.get("REQUESTED_MODEL", "")
requested_base = requested.rstrip("/").split("/")[-1] if requested else ""

try:
    payload = json.loads(response)
except Exception:
    print("")
    raise SystemExit(0)

data = payload.get("data", [])
model_ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]

if not model_ids:
    print("")
elif requested and requested in model_ids:
    print(requested)
elif requested_base and requested_base in model_ids:
    print(requested_base)
else:
    print(model_ids[0])
PY
    )

    if [ -z "$detected_model" ]; then
        return 1
    fi

    JUDGE_API_BASE="$api_base"
    JUDGE_MODEL="$detected_model"
    LOCAL_JUDGE_MODE=false
    echo "[INFO] 发现已启动的本地 vLLM Judge，直接复用: api_base=$JUDGE_API_BASE | model=$JUDGE_MODEL"
    return 0
}

benchmarks_need_llm_judge() {
    # Check if any benchmark in the list needs an LLM judge
    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    local needs_judge=false
    for bench in "${BENCH_ARRAY[@]}"; do
        bench="$(echo "$bench" | tr -d ' ')"
        case "$bench" in
            alpacaeval)
                needs_judge=true
                ;;
        esac
    done
    $needs_judge || return 1

    # If skip-existing is set, check whether all models already have alpacaeval results.
    # If so, the judge is not actually needed this run.
    if [ "$SKIP_EXISTING" = true ]; then
        local model_ids
        model_ids=$(get_model_ids "$MODEL_INPUT")
        local any_needs_eval=false
        for mid in $model_ids; do
            if [ ! -f "$ROOT_DIR/external_evals/alpaca_eval/results/$mid/leaderboard.csv" ]; then
                any_needs_eval=true
                break
            fi
        done
        $any_needs_eval || return 1
    fi

    return 0
}

stop_local_judge_vllm() {
    if [ -n "$LOCAL_JUDGE_VLLM_PID" ]; then
        echo "[INFO] 停止本地 Judge vLLM 服务..."
        kill "$LOCAL_JUDGE_VLLM_PID" 2>/dev/null || true
        wait "$LOCAL_JUDGE_VLLM_PID" 2>/dev/null || true
    fi
}

start_local_judge_vllm() {
    GPU_IDS=()
    detect_gpu_ids GPU_IDS
    GPU_COUNT=${#GPU_IDS[@]}

    if [ "$GPU_COUNT" -lt 1 ]; then
        echo "[ERROR] 未检测到可用 GPU，无法启动本地 vLLM Judge"
        exit 1
    fi

    local tp="$JUDGE_TP_SIZE"
    if ! [[ "$tp" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --judge-tp-size 必须是非负整数: $tp"
        exit 1
    fi
    if [ "$tp" -eq 0 ]; then
        tp="$GPU_COUNT"
    fi
    if [ "$tp" -gt "$GPU_COUNT" ]; then
        echo "[ERROR] judge TP_SIZE($tp) > 可用 GPU 数($GPU_COUNT)"
        exit 1
    fi

    if ! [[ "$JUDGE_PORT" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --judge-port 必须是整数: $JUDGE_PORT"
        exit 1
    fi

    LOCAL_JUDGE_MODEL_API_NAME="$(basename "$JUDGE_MODEL")"
    if [ -z "$LOCAL_JUDGE_MODEL_API_NAME" ] || [ "$LOCAL_JUDGE_MODEL_API_NAME" = "." ] || [ "$LOCAL_JUDGE_MODEL_API_NAME" = "/" ]; then
        LOCAL_JUDGE_MODEL_API_NAME="local_judge_model"
    fi
    local cuda_ids
    cuda_ids="$(join_by , "${GPU_IDS[@]}")"
    local log_file="$TOOLS_DIR/vllm_judge_$(sanitize_for_filename "$LOCAL_JUDGE_MODEL_API_NAME")_p${JUDGE_PORT}.log"

    echo "[INFO] 启动本地 vLLM Judge: model=$JUDGE_MODEL | served_name=$LOCAL_JUDGE_MODEL_API_NAME | cuda=$cuda_ids | tp=$tp | port=$JUDGE_PORT"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] would start local judge vLLM -> $log_file"
        JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1"
        JUDGE_MODEL="$LOCAL_JUDGE_MODEL_API_NAME"
        LOCAL_JUDGE_MODE=true
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="$cuda_ids" conda run -n vllm python -m vllm.entrypoints.openai.api_server \
            --model "$JUDGE_MODEL" \
            --served-model-name "$LOCAL_JUDGE_MODEL_API_NAME" \
            --port "$JUDGE_PORT" \
            --gpu-memory-utilization "$JUDGE_GPU_MEM" \
            --tensor-parallel-size "$tp" \
            --trust-remote-code \
            > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$cuda_ids" python -m vllm.entrypoints.openai.api_server \
            --model "$JUDGE_MODEL" \
            --served-model-name "$LOCAL_JUDGE_MODEL_API_NAME" \
            --port "$JUDGE_PORT" \
            --gpu-memory-utilization "$JUDGE_GPU_MEM" \
            --tensor-parallel-size "$tp" \
            --trust-remote-code \
            > "$log_file" 2>&1 &
    fi
    LOCAL_JUDGE_VLLM_PID=$!
    LOCAL_JUDGE_MODE=true

    if ! wait_for_server "$JUDGE_PORT"; then
        echo "[ERROR] 本地 Judge vLLM 启动失败，日志: $log_file"
        stop_local_judge_vllm
        exit 1
    fi

    JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1"
    JUDGE_MODEL="$LOCAL_JUDGE_MODEL_API_NAME"
}

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
# 可选第三参数 bench_list 覆盖全局 BENCHMARKS
generate_config() {
    local model_id=$1
    local config_file=$2
    local bench_list="${3:-$BENCHMARKS}"

    # 构建 benchmark 配置
    local bench_yaml=""
    IFS=',' read -ra BENCH_ARRAY <<< "$bench_list"
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

# 运行单个模型的非 judge 类评估 (IFEval / TruthfulQA / LiveBench)
# 参数: model_id  bench_list
run_model_evaluate() {
    local model_id=$1
    local bench_list="$2"   # 必传，已去掉 alpacaeval

    # 如果列表为空就不做任何事
    [ -z "$bench_list" ] && return 0

    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVALUATE (no-judge): $model_id"
    echo "=========================================================="

    # 检查每个 benchmark 的生成结果 & skip-existing
    local has_work=false
    IFS=',' read -ra BENCH_ARRAY <<< "$bench_list"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | tr -d ' ')
        case $bench in
            ifeval)
                if [ "$SKIP_EXISTING" = true ] && \
                   [ -f "$RESULTS_DIR/$model_id/ifeval/eval_results_strict.jsonl" ]; then
                    echo "[INFO] [$model_id] IFEval 已存在，跳过"
                elif [ -f "$RESULTS_DIR/$model_id/ifeval/candidate_outputs.jsonl" ]; then
                    has_work=true
                fi
                ;;
            truthfulqa)
                if [ "$SKIP_EXISTING" = true ] && \
                   [ -f "$RESULTS_DIR/$model_id/truthfulqa/results.csv" ]; then
                    echo "[INFO] [$model_id] TruthfulQA 已存在，跳过"
                elif [ -f "$RESULTS_DIR/$model_id/truthfulqa/TruthfulQA_generated.csv" ]; then
                    has_work=true
                fi
                ;;
            livebench)
                if [ "$SKIP_EXISTING" = true ] && \
                   [ "$(find "$RESULTS_DIR/$model_id/livebench" -name '*.jsonl' 2>/dev/null | wc -l)" -gt 0 ]; then
                    echo "[INFO] [$model_id] LiveBench 已存在，跳过"
                else
                    has_work=true
                fi
                ;;
        esac
    done

    if [ "$has_work" = false ]; then
        echo "[INFO] [$model_id] no-judge 阶段无需执行，跳过"
        return 0
    fi

    local config_file="$TOOLS_DIR/config_eval_${model_id}.yaml"
    generate_config "$model_id" "$config_file" "$bench_list"

    local eval_args=("$config_file" --phase evaluate)
    [ "$SKIP_EXISTING" = true ] && eval_args+=(--skip-existing)

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] would run: python $TOOLS_DIR/run_evals.py ${eval_args[*]}"
    elif python "$TOOLS_DIR/run_evals.py" "${eval_args[@]}"; then
        echo "[INFO] ✓ [$model_id] no-judge 评估完成"
    else
        echo "[ERROR] ✗ [$model_id] no-judge 评估失败"
    fi

    rm -f "$config_file"

    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成 (no-judge): $model_id"
    echo "=========================================================="
}

# 运行单个模型的 AlpacaEval2 (需要 judge 已就绪)
run_model_alpacaeval() {
    local model_id=$1

    local alpacaeval_out="$ROOT_DIR/external_evals/alpaca_eval/results/$model_id/leaderboard.csv"
    if [ "$SKIP_EXISTING" = true ] && [ -f "$alpacaeval_out" ]; then
        echo "[INFO] [$model_id] AlpacaEval2 已存在，跳过"
        return 0
    fi

    if [ ! -f "$RESULTS_DIR/$model_id/alpacaeval2/model_outputs.json" ]; then
        echo "[WARN] [$model_id] 未找到 AlpacaEval2 生成结果，跳过"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] AlpacaEval2: $model_id"

    local config_file="$TOOLS_DIR/config_eval_alpaca_${model_id}.yaml"
    generate_config "$model_id" "$config_file" "alpacaeval"

    local eval_args=("$config_file" --phase evaluate)
    [ "$SKIP_EXISTING" = true ] && eval_args+=(--skip-existing)

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] would run alpacaeval: python $TOOLS_DIR/run_evals.py ${eval_args[*]}"
    elif python "$TOOLS_DIR/run_evals.py" "${eval_args[@]}"; then
        echo "[INFO] ✓ [$model_id] AlpacaEval2 完成"
    else
        echo "[ERROR] ✗ [$model_id] AlpacaEval2 失败"
    fi

    rm -f "$config_file"
}

# 获取模型 ID 列表
get_model_ids() {
    local input=$1

    if [ ! -d "$input" ]; then
        echo "$input"
        return
    fi

    if [[ "$input" == *"results"* ]]; then
        ls -1 "$input" 2>/dev/null
        return
    fi

    mapfile -t subdirs < <(find "$input" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    visible_subdirs=()
    for model_path in "${subdirs[@]}"; do
        child_name="$(basename "$model_path")"
        if [[ "$child_name" == .* ]]; then
            continue
        fi
        visible_subdirs+=("$model_path")
    done

    checkpoint_subdirs=()
    for model_path in "${visible_subdirs[@]}"; do
        child_name="$(basename "$model_path")"
        if [[ "$child_name" == checkpoint* ]]; then
            checkpoint_subdirs+=("$model_path")
        fi
    done

    if [ ${#checkpoint_subdirs[@]} -gt 0 ]; then
        parent_model_name="$(basename "$input")"
        for model_path in "${checkpoint_subdirs[@]}"; do
            checkpoint_name="$(basename "$model_path")"
            echo "${parent_model_name}_${checkpoint_name}"
        done
        echo "$parent_model_name"
        return
    fi

    if [ ${#visible_subdirs[@]} -gt 0 ]; then
        for model_path in "${visible_subdirs[@]}"; do
            echo "$(basename "$model_path")"
        done
        return
    fi

    echo "$(basename "$input")"
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
echo "  Judge Port:     $JUDGE_PORT"
echo "  Judge TP Size:  $JUDGE_TP_SIZE"
echo "  Skip Existing:  $SKIP_EXISTING"
echo "  Dry Run:        $DRY_RUN"
echo ""

# 获取模型 ID 列表
MODEL_IDS=$(get_model_ids "$MODEL_INPUT")

if [ -z "$MODEL_IDS" ]; then
    echo "[ERROR] 没有找到任何模型"
    exit 1
fi

MODEL_COUNT=$(echo "$MODEL_IDS" | wc -l)
echo "[INFO] 找到 $MODEL_COUNT 个模型"
echo ""

# ---- 拆分 benchmark 列表 ----
NO_JUDGE_BENCHES=$(echo "$BENCHMARKS" | tr ',' '\n' | grep -v '^alpacaeval$' | paste -sd ',' -)
HAS_ALPACAEVAL=false
echo "$BENCHMARKS" | tr ',' '\n' | grep -qx 'alpacaeval' && HAS_ALPACAEVAL=true

# ==================== PHASE 1: 非 judge 类 benchmarks ====================
if [ -n "$NO_JUDGE_BENCHES" ]; then
    echo ""
    echo "============================================================"
    echo "  PHASE 1/2 — No-judge benchmarks: $NO_JUDGE_BENCHES"
    echo "============================================================"
    for model_id in $MODEL_IDS; do
        run_model_evaluate "$model_id" "$NO_JUDGE_BENCHES"
    done
fi

# ==================== PHASE 2: AlpacaEval (需要 judge) ====================
if [ "$HAS_ALPACAEVAL" = true ]; then
    # 检查是否有任何模型还需要跑 AlpacaEval
    any_needs_alpacaeval=false
    for model_id in $MODEL_IDS; do
        if [ "$SKIP_EXISTING" = false ] || \
           [ ! -f "$ROOT_DIR/external_evals/alpaca_eval/results/$model_id/leaderboard.csv" ]; then
            if [ -f "$RESULTS_DIR/$model_id/alpacaeval2/model_outputs.json" ]; then
                any_needs_alpacaeval=true
                break
            fi
        fi
    done

    echo ""
    echo "============================================================"
    echo "  PHASE 2/2 — AlpacaEval2 (judge needed)"
    echo "============================================================"

    if [ "$any_needs_alpacaeval" = false ]; then
        echo "[INFO] 所有模型的 AlpacaEval2 结果已存在，跳过 judge 启动"
    else
        if should_use_local_vllm_judge; then
            if try_reuse_existing_local_judge; then
                echo "[INFO] 复用已启动 Judge，跳过新服务启动"
            else
                start_local_judge_vllm
                trap 'stop_local_judge_vllm' EXIT
                echo "[INFO] 使用本地 vLLM Judge: api_base=$JUDGE_API_BASE | model=$JUDGE_MODEL"
            fi
        fi
        for model_id in $MODEL_IDS; do
            run_model_alpacaeval "$model_id"
        done
    fi
fi

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
