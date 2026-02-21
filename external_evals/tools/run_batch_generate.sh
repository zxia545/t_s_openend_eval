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

# ==================== 帮助信息 ====================
show_help() {
    cat << EOF
Flexible Batch Generation Script for LLM Evaluation

用法:
    $0 <model_input> [options]

参数:
    model_input          模型输入 (必需)
                         - 文件夹: 遍历所有子目录作为模型
                         - HuggingFace ID: 直接使用 (如 Qwen/Qwen2.5-0.5B-Instruct)
                         - 本地路径: 使用单个模型

选项:
    -p, --port PORT      vLLM 服务端口 (默认: $DEFAULT_PORT)
    -g, --gpu-mem UTIL   GPU 内存使用率 (默认: $DEFAULT_GPU_MEM)
    -t, --tp-size SIZE   Tensor Parallel 大小 (默认: $DEFAULT_TP_SIZE)
    -b, --benchmarks LIST  要运行的 benchmarks (逗号分隔)
                         (默认: $DEFAULT_BENCHMARKS)
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
source activate vllm
conda activate vllm

# ==================== 辅助函数 ====================

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
    
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GENERATION: $model_id"
    echo "=========================================================="
    
    # 检查是否跳过
    if [ "$SKIP_EXISTING" = true ] && check_existing_results "$model_id"; then
        echo "[INFO] 结果已存在，跳过: $model_id"
        return 0
    fi
    
    # 启动 vLLM 服务
    echo "[INFO] 启动 vLLM 服务 (端口: $PORT, GPU: $GPU_MEM, TP: $TP_SIZE)..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$model_id" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM" \
        --tensor-parallel-size "$TP_SIZE" \
        --trust-remote-code \
        > "$TOOLS_DIR/vllm_${model_id}.log" 2>&1 &
        
    local vllm_pid=$!
    echo "[INFO] vLLM PID: $vllm_pid | 日志: $TOOLS_DIR/vllm_${model_id}.log"
    
    # 等待服务就绪
    if ! wait_for_server "$PORT"; then
        echo "[ERROR] vLLM 启动失败: $model_id"
        kill -9 $vllm_pid 2>/dev/null || true
        return 1
    fi
    
    # 生成配置
    local config_file="$TOOLS_DIR/config_${model_id}.yaml"
    generate_config "$model_id" "http://localhost:${PORT}/v1" "$config_file"
    
    # 运行生成
    echo "[INFO] 开始生成..."
    if python "$TOOLS_DIR/run_evals.py" "$config_file" --phase generate; then
        echo "[INFO] ✓ 生成完成: $model_id"
    else
        echo "[ERROR] ✗ 生成失败: $model_id"
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
echo ""

# 判断输入类型
if [[ "$MODEL_INPUT" == *"/"* ]] && [[ ! "$MODEL_INPUT" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$ ]]; then
    # 本地路径
    if [ -d "$MODEL_INPUT" ]; then
        # 检查是否是 checkpoints 文件夹 (包含子目录)
        local subdirs=($(find "$MODEL_INPUT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null))
        if [ ${#subdirs[@]} -gt 0 ]; then
            # 多个 checkpoint
            echo "[INFO] 检测到 checkpoints 文件夹，共 ${#subdirs[@]} 个模型"
            for model_path in "${subdirs[@]}"; do
                model_id=$(basename "$model_path")
                run_model_generate "$model_path" "$model_id"
            done
        else
            # 单个模型
            model_id=$(basename "$MODEL_INPUT")
            run_model_generate "$MODEL_INPUT" "$model_id"
        fi
    else
        echo "[ERROR] 路径不存在: $MODEL_INPUT"
        exit 1
    fi
else
    # HuggingFace 模型 ID 或单个名称
    model_id=$(basename "$MODEL_INPUT")
    run_model_generate "$MODEL_INPUT" "$model_id"
fi

echo ""
echo "============================================================"
echo "            所有生成任务完成!"
echo "============================================================"
echo ""
echo "结果位置: $RESULTS_DIR/"
