#!/usr/bin/env bash
#
# Flexible Batch Generation Script
# 
# 用法:
#   ./run_batch_generate.sh <model_input> [options]
#
# Model Input 类型:
#   1. 文件夹路径: /path/to/checkpoints  (遍历所有子目录)
#   2. HuggingFace 模型: Qwen/Qwen2.5-0.5B-Instruct
#   3. 本地模型路径: /root/models/my-model
#   4. 单个 checkpoint: /path/to/checkpoints/step-1000
#
# 示例:
#   # 评估 SFT 训练产生的多个 checkpoint
#   ./run_batch_generate.sh /root/checkpoints
#
#   # 评估单个 HuggingFace 模型
#   ./run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct
#
#   # 评估本地模型，指定端口和 GPU 参数
#   ./run_batch_generate.sh /root/models/my-model --port 8002 --gpu-mem 0.9
#
#   # 只生成特定的 benchmarks
#   ./run_batch_generate.sh Qwen/Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa
#

set -euo pipefail

# ==================== 默认配置 ====================
DEFAULT_PORT=8001
DEFAULT_GPU_MEM=0.85
DEFAULT_TP_SIZE=1
DEFAULT_BENCHMARKS="ifeval,truthfulqa,alpacaeval,livebench"
DEFAULT_VLLM_CONDA_ENV="vllm"
DEFAULT_STAGGER_SECONDS=0

# ==================== 帮助信息 ====================
show_help() {
    cat << EOF
Flexible Batch Generation Script for LLM Evaluation

用法:
    $0 <model_input> [options]

参数:
    model_input          模型输入 (必需)
                          - 文件夹: 遍历所有子目录作为模型
                          - 若存在 checkpoint* 子目录: 运行所有 checkpoint*，并额外运行该目录本身(视为 final model)
                          - HuggingFace ID: 直接使用 (如 Qwen/Qwen2.5-0.5B-Instruct)
                          - 本地路径: 使用单个模型

选项:
    -p, --port PORT      vLLM 服务端口 (默认: $DEFAULT_PORT)
    -g, --gpu-mem UTIL   GPU 内存使用率 (默认: $DEFAULT_GPU_MEM)
    -t, --tp-size SIZE   Tensor Parallel 大小 (默认: $DEFAULT_TP_SIZE)
    -b, --benchmarks LIST  要运行的 benchmarks (逗号分隔)
                          (默认: $DEFAULT_BENCHMARKS)
    --gpus LIST          使用指定 GPU (逗号分隔，如 0,1,2,3；默认: 自动探测)
    --num-workers N      并行 worker 数量 (默认: 自动 = floor(num_gpus / tp_size))
    --stagger-seconds S  每个 worker 启动前错峰等待秒数 (默认: $DEFAULT_STAGGER_SECONDS)
    --no-parallel        强制串行 (默认: 自动并行)
    --dry-run            只打印调度计划，不实际启动 vLLM/跑生成
    --skip-existing      跳过已有结果的模型
    -h, --help           显示此帮助信息

示例:
    # 评估 checkpoints 文件夹中的所有模型
    $0 /path/to/checkpoints

    # 评估单个 HuggingFace 模型
    $0 Qwen/Qwen2.5-0.5B-Instruct

    # 自定义端口和 GPU
    $0 /path/to/model --port 8002 --gpu-mem 0.9

    # 只运行特定 benchmarks
    $0 Qwen/Qwen2.5-0.5B-Instruct --benchmarks ifeval,truthfulqa

EOF
}

# ==================== 参数解析 ====================
MODEL_INPUT=""
PORT=$DEFAULT_PORT
GPU_MEM=$DEFAULT_GPU_MEM
TP_SIZE=$DEFAULT_TP_SIZE
BENCHMARKS=$DEFAULT_BENCHMARKS
SKIP_EXISTING=false
GPU_IDS_CSV=""
NUM_WORKERS=0
STAGGER_SECONDS=$DEFAULT_STAGGER_SECONDS
PARALLEL_MODE="auto"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -g|--gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        -t|--tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        -b|--benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS_CSV="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --stagger-seconds)
            STAGGER_SECONDS="$2"
            shift 2
            ;;
        --no-parallel)
            PARALLEL_MODE="off"
            shift
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

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$DEFAULT_VLLM_CONDA_ENV" || {
        echo "[ERROR] conda env 激活失败: $DEFAULT_VLLM_CONDA_ENV"
        exit 1
    }
else
    source activate "$DEFAULT_VLLM_CONDA_ENV" 2>/dev/null || true
fi

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
        out_arr[$i]="$(echo "${out_arr[$i]}" | tr -d ' ' )"
    done
}

detect_gpu_ids() {
    local out_name="$1"
    local -n out_arr="$out_name"

    if [ -n "$GPU_IDS_CSV" ]; then
        split_csv_to_array "$GPU_IDS_CSV" "$out_name"
        return 0
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
        split_csv_to_array "$CUDA_VISIBLE_DEVICES" "$out_name"
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

# 等待 vLLM 服务就绪
wait_for_server() {
    local port=$1
    local max_retries=120
    local retry=0
    
    echo "等待 vLLM 服务在端口 $port 启动..."
    while [ $retry -lt $max_retries ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "✓ 服务已就绪!"
            return 0
        fi
        sleep 5
        retry=$((retry+1))
        printf "."
    done
    
    echo ""
    echo "✗ 等待服务超时!"
    return 1
}

# 检查是否已有生成结果
check_existing_results() {
    local model_id=$1
    local has_results=true
    
    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | tr -d ' ')
        case $bench in
            ifeval)
                [ -f "$RESULTS_DIR/$model_id/ifeval/candidate_outputs.jsonl" ] || has_results=false
                ;;
            truthfulqa)
                [ -f "$RESULTS_DIR/$model_id/truthfulqa/TruthfulQA_generated.csv" ] || has_results=false
                ;;
            alpacaeval)
                [ -f "$RESULTS_DIR/$model_id/alpacaeval2/model_outputs.json" ] || has_results=false
                ;;
            livebench)
                [ -d "$ROOT_DIR/external_evals/livebench/livebench/model_answers/$model_id" ] || has_results=false
                ;;
        esac
    done
    
    $has_results
}

# 生成 YAML 配置
generate_config() {
    local model_id=$1
    local api_base=$2
    local config_file=$3
    
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
    api_base: "${api_base}"
    api_key: "dummy"

benchmarks:
$(echo -e "$bench_yaml")
EOF
}

# 运行单个模型的生成
run_model_generate() {
    local model_path=$1
    local model_id=$2
    local port=$3
    local cuda_visible_devices=$4
    local worker_label=${5:-""}

    local log_model_id
    log_model_id="$(sanitize_for_filename "$model_id")"
    
    echo "=========================================================="
    if [ -n "$worker_label" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GENERATION: $model_id | worker=$worker_label | port=$port | cuda=$cuda_visible_devices"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GENERATION: $model_id | port=$port | cuda=$cuda_visible_devices"
    fi
    echo "=========================================================="
    
    # 检查是否跳过
    if [ "$SKIP_EXISTING" = true ] && check_existing_results "$model_id"; then
        echo "[INFO] 结果已存在，跳过: $model_id"
        return 0
    fi
    
    # 启动 vLLM 服务
    echo "[INFO] 启动 vLLM 服务 (端口: $port, GPU_MEM: $GPU_MEM, TP: $TP_SIZE, CUDA_VISIBLE_DEVICES: $cuda_visible_devices)..."
    local vllm_log="$TOOLS_DIR/vllm_${log_model_id}_p${port}.log"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] would start vLLM -> $vllm_log"
        echo "[DRY-RUN] would run evals via port $port"
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$model_id" \
        --port "$port" \
        --gpu-memory-utilization "$GPU_MEM" \
        --tensor-parallel-size "$TP_SIZE" \
        --trust-remote-code \
        > "$vllm_log" 2>&1 &
        
    local vllm_pid=$!
    echo "[INFO] vLLM PID: $vllm_pid | 日志: $vllm_log"
    
    # 等待服务就绪
    if ! wait_for_server "$port"; then
        echo "[ERROR] vLLM 启动失败: $model_id"
        kill -9 $vllm_pid 2>/dev/null || true
        return 1
    fi
    
    # 生成配置
    local config_file="$TOOLS_DIR/config_${log_model_id}_p${port}.yaml"
    generate_config "$model_id" "http://localhost:${port}/v1" "$config_file"
    
    # 运行生成
    echo "[INFO] 开始生成..."
    if python "$TOOLS_DIR/run_evals.py" "$config_file" --phase generate; then
        echo "[INFO] ✓ 生成完成: $model_id"
    else
        echo "[ERROR] ✗ 生成失败: $model_id"
        echo "[INFO] 停止 vLLM 服务..."
        kill $vllm_pid 2>/dev/null || true
        wait $vllm_pid 2>/dev/null || true
        rm -f "$config_file"
        return 1
    fi
    
    # 停止服务
    echo "[INFO] 停止 vLLM 服务..."
    kill $vllm_pid 2>/dev/null || true
    wait $vllm_pid 2>/dev/null || true
    
    # 清理
    rm -f "$config_file"
    
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: $model_id"
    echo "=========================================================="
    
    sleep 5
}

launch_model_in_slot() {
    local slot_idx=$1
    local item=$2
    local is_first_launch=$3

    local model_path
    local model_id
    IFS=$'\t' read -r model_path model_id <<< "$item"

    local slot_port="${SLOT_PORTS[$slot_idx]}"
    local slot_cuda="${SLOT_CUDAS[$slot_idx]}"
    local delay=0
    if [ "$is_first_launch" = true ] && [ "$STAGGER_SECONDS" -gt 0 ]; then
        delay=$((slot_idx * STAGGER_SECONDS))
    fi

    echo "[INFO] 分配任务到 slot=$slot_idx | port=$slot_port | cuda=$slot_cuda | model=$model_id"
    (
        if [ "$delay" -gt 0 ]; then
            echo "[INFO] slot=$slot_idx 将等待 $delay 秒后开始 (stagger)"
            sleep "$delay"
        fi
        run_model_generate "$model_path" "$model_id" "$slot_port" "$slot_cuda" "$slot_idx"
    ) &
    local job_pid=$!
    PID_TO_SLOT[$job_pid]="$slot_idx"
}

# ==================== 主逻辑 ====================

echo "============================================================"
echo "            LLM Evaluation - Generation Phase"
echo "============================================================"
echo ""
echo "配置:"
echo "  Model Input:    $MODEL_INPUT"
echo "  Port:           $PORT"
echo "  GPU Memory:     $GPU_MEM"
echo "  TP Size:        $TP_SIZE"
echo "  Benchmarks:     $BENCHMARKS"
echo "  Skip Existing:  $SKIP_EXISTING"
echo "  GPUs:           ${GPU_IDS_CSV:-auto}"
echo "  Workers:        ${NUM_WORKERS:-auto}"
echo "  Stagger(s):     $STAGGER_SECONDS"
echo "  Parallel:       $PARALLEL_MODE"
echo "  Dry Run:        $DRY_RUN"
echo ""

MODELS=()
add_model() {
    local model_path=$1
    local model_id=$2
    MODELS+=("$model_path"$'\t'"$model_id")
}

if [[ "$MODEL_INPUT" == *"/"* ]] && [[ ! "$MODEL_INPUT" =~ ^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$ ]]; then
    if [ -d "$MODEL_INPUT" ]; then
        mapfile -t subdirs < <(find "$MODEL_INPUT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
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
            echo "[INFO] 检测到 checkpoint 子目录，共 ${#checkpoint_subdirs[@]} 个；同时将输入目录作为 final model 运行"
            for model_path in "${checkpoint_subdirs[@]}"; do
                add_model "$model_path" "$(basename "$model_path")"
            done
            add_model "$MODEL_INPUT" "$(basename "$MODEL_INPUT")"
        elif [ ${#visible_subdirs[@]} -gt 0 ]; then
            echo "[INFO] 检测到模型目录，共 ${#visible_subdirs[@]} 个"
            for model_path in "${visible_subdirs[@]}"; do
                add_model "$model_path" "$(basename "$model_path")"
            done
        else
            add_model "$MODEL_INPUT" "$(basename "$MODEL_INPUT")"
        fi
    else
        echo "[ERROR] 路径不存在: $MODEL_INPUT"
        exit 1
    fi
else
    add_model "$MODEL_INPUT" "$(basename "$MODEL_INPUT")"
fi

MODEL_COUNT=${#MODELS[@]}
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "[ERROR] 未找到任何模型"
    exit 1
fi

GPU_IDS=()
detect_gpu_ids GPU_IDS
VALID_GPU_IDS=()
for gid in "${GPU_IDS[@]}"; do
    if [ -z "$gid" ]; then
        continue
    fi
    if [[ "$gid" =~ ^[0-9]+$ ]]; then
        VALID_GPU_IDS+=("$gid")
    else
        echo "[ERROR] 非法 GPU id: '$gid' (期望整数，如 0,1,2)"
        exit 1
    fi
done
if [ ${#VALID_GPU_IDS[@]} -eq 0 ]; then
    VALID_GPU_IDS=("0")
fi
GPU_IDS=("${VALID_GPU_IDS[@]}")
GPU_COUNT=${#GPU_IDS[@]}

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --num-workers 必须是非负整数: $NUM_WORKERS"
    exit 1
fi
if ! [[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --stagger-seconds 必须是非负整数: $STAGGER_SECONDS"
    exit 1
fi

if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]] || [ "$TP_SIZE" -le 0 ]; then
    echo "[ERROR] --tp-size 必须是正整数: $TP_SIZE"
    exit 1
fi
if [ "$TP_SIZE" -gt "$GPU_COUNT" ]; then
    echo "[ERROR] TP_SIZE($TP_SIZE) > 可用 GPU 数($GPU_COUNT)。可用 GPU: $(join_by , "${GPU_IDS[@]}")"
    exit 1
fi

MAX_WORKERS=$((GPU_COUNT / TP_SIZE))
if [ "$MAX_WORKERS" -lt 1 ]; then
    MAX_WORKERS=1
fi

WORKERS=1
if [ "$PARALLEL_MODE" != "off" ] && [ "$MODEL_COUNT" -gt 1 ] && [ "$MAX_WORKERS" -gt 1 ]; then
    WORKERS="$MAX_WORKERS"
fi
if [ "$NUM_WORKERS" -gt 0 ]; then
    WORKERS="$NUM_WORKERS"
fi

if [ "$WORKERS" -gt "$MAX_WORKERS" ]; then
    WORKERS="$MAX_WORKERS"
fi
if [ "$WORKERS" -gt "$MODEL_COUNT" ]; then
    WORKERS="$MODEL_COUNT"
fi
if [ "$WORKERS" -lt 1 ]; then
    WORKERS=1
fi

echo "[INFO] models=$MODEL_COUNT | gpus=$GPU_COUNT($(join_by , "${GPU_IDS[@]}") ) | tp=$TP_SIZE | workers=$WORKERS | base_port=$PORT | stagger=$STAGGER_SECONDS"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] 调度计划:"
    for ((w=0; w<WORKERS; w++)); do
        if [ "$TP_SIZE" -eq 1 ]; then
            cuda_str="${GPU_IDS[$((w % GPU_COUNT))]}"
        else
            start=$((w * TP_SIZE))
            gpu_slice=("${GPU_IDS[@]:$start:$TP_SIZE}")
            cuda_str="$(join_by , "${gpu_slice[@]}")"
        fi
        echo "  worker=$w port=$((PORT + w)) cuda=$cuda_str"
    done
    echo "[DRY-RUN] 模型列表:"
    for i in "${!MODELS[@]}"; do
        IFS=$'\t' read -r mp mid <<< "${MODELS[$i]}"
        echo "  [$i] $mid -> $mp"
    done
    exit 0
fi

EXIT_STATUS=0

if [ "$WORKERS" -le 1 ]; then
    cuda_str="$(join_by , "${GPU_IDS[@]:0:$TP_SIZE}")"
    for item in "${MODELS[@]}"; do
        IFS=$'\t' read -r model_path model_id <<< "$item"
        if ! run_model_generate "$model_path" "$model_id" "$PORT" "$cuda_str"; then
            EXIT_STATUS=1
        fi
    done
else
    SLOT_CUDAS=()
    SLOT_PORTS=()
    for ((w=0; w<WORKERS; w++)); do
        if [ "$TP_SIZE" -eq 1 ]; then
            cuda_str="${GPU_IDS[$w]}"
        else
            start=$((w * TP_SIZE))
            gpu_slice=("${GPU_IDS[@]:$start:$TP_SIZE}")
            cuda_str="$(join_by , "${gpu_slice[@]}")"
        fi
        port=$((PORT + w))
        SLOT_CUDAS+=("$cuda_str")
        SLOT_PORTS+=("$port")
        echo "[INFO] 初始化 slot=$w | port=$port | cuda=$cuda_str"
    done

    pending_idx=0
    declare -A PID_TO_SLOT=()

    for ((slot=0; slot<WORKERS; slot++)); do
        if [ "$pending_idx" -ge "$MODEL_COUNT" ]; then
            break
        fi
        launch_model_in_slot "$slot" "${MODELS[$pending_idx]}" true
        pending_idx=$((pending_idx + 1))
    done

    while [ ${#PID_TO_SLOT[@]} -gt 0 ]; do
        finished_pid=""
        if wait -n -p finished_pid; then
            wait_rc=0
        else
            wait_rc=$?
        fi

        if [ -z "$finished_pid" ]; then
            continue
        fi
        finished_slot="${PID_TO_SLOT[$finished_pid]}"
        unset 'PID_TO_SLOT[$finished_pid]'

        if [ "$wait_rc" -ne 0 ]; then
            EXIT_STATUS=1
        fi

        if [ "$pending_idx" -lt "$MODEL_COUNT" ]; then
            launch_model_in_slot "$finished_slot" "${MODELS[$pending_idx]}" false
            pending_idx=$((pending_idx + 1))
        fi
    done
fi

if [ "$EXIT_STATUS" -ne 0 ]; then
    echo "[ERROR] 部分模型生成失败 (exit=$EXIT_STATUS)"
    exit "$EXIT_STATUS"
fi

echo ""
echo "============================================================"
echo "            所有生成任务完成!"
echo "============================================================"
echo ""
echo "结果位置: $RESULTS_DIR/"
