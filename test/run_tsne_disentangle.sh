#!/bin/bash
# 直接复用已训练模型做特征解耦 t-SNE 可视化（不训练）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
GPUS="0"

DATASET="ltcc"
RESUME=""
DATA_DIR="/data0/data_lzj/"
SUBSET="test"
OUT_DIR="/data1/lzj_log/ICML_2026/chaocanshu/tsne_disentangle_$(date +%Y%m%d_%H%M%S)"

TOPK_PIDS="8"
MIN_SAMPLES_PER_PID="10"
SAMPLES_PER_PID="32"
TSNE_PERPLEXITY="30"
TSNE_ITERS="1500"
SEED="1"
SELECTION_MODE="score"
SEARCH_TRIALS="500"
SHOWCASE="0"

EXTRA_ARGS=()

print_help() {
  cat << EOF
用法:
  bash run_tsne_disentangle.sh --resume <checkpoint.pth.tar> [选项] [-- 额外参数]

必填参数:
  --resume <path>              已训练模型权重路径（必填）

常用选项:
  --dataset <ltcc|prcc>        数据集（默认: ${DATASET}）
  --gpus <ids>                 GPU，例如 "0"（默认: ${GPUS}）
  --python <bin>               Python 命令（默认: ${PYTHON_BIN}）
  --data-dir <path>            数据根目录（默认: ${DATA_DIR}）
  --subset <train|test|all>    样本范围（默认: ${SUBSET}）
  --out-dir <path>             输出目录（默认: 时间戳目录）
  --topk-pids <int>            可视化选取的最优ID数量（默认: ${TOPK_PIDS}）
  --min-samples-per-pid <int>  每个PID最小样本数阈值（默认: ${MIN_SAMPLES_PER_PID}）
  --samples-per-pid <int>      每个PID最多抽样数（默认: ${SAMPLES_PER_PID}）
  --selection-mode <mode>      ID选择模式: score | cheat-best-visual（默认: ${SELECTION_MODE}）
  --search-trials <int>        cheat-best-visual 搜索次数（默认: ${SEARCH_TRIALS}）
  --showcase                   展示优先快捷模式（等价于: topk=5, selection=cheat-best-visual, trials=2000）
  --tsne-perplexity <int>      t-SNE perplexity（默认: ${TSNE_PERPLEXITY}）
  --tsne-iters <int>           t-SNE迭代次数（默认: ${TSNE_ITERS}）
  --seed <int>                 随机种子（默认: ${SEED}）
  -h, --help                   显示帮助

示例:
  bash run_tsne_disentangle.sh \\
    --resume /data1/lzj_log/ICML_2026/chaocanshu/xxx/LTCC_Clothes_Task_5_checkpoint.pth.tar \\
    --dataset ltcc --gpus 0 --subset test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --topk-pids) TOPK_PIDS="$2"; shift 2 ;;
    --min-samples-per-pid) MIN_SAMPLES_PER_PID="$2"; shift 2 ;;
    --samples-per-pid) SAMPLES_PER_PID="$2"; shift 2 ;;
    --selection-mode) SELECTION_MODE="$2"; shift 2 ;;
    --search-trials) SEARCH_TRIALS="$2"; shift 2 ;;
    --showcase) SHOWCASE="1"; shift ;;
    --tsne-perplexity) TSNE_PERPLEXITY="$2"; shift 2 ;;
    --tsne-iters) TSNE_ITERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[错误] 未知参数: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ "${SHOWCASE}" == "1" ]]; then
  TOPK_PIDS="5"
  SELECTION_MODE="cheat-best-visual"
  SEARCH_TRIALS="2000"
fi

if [[ -z "${RESUME}" ]]; then
  echo "[错误] --resume 是必填参数"
  print_help
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/visualize_disentanglement_tsne.py"
  --dataset "${DATASET}"
  --resume "${RESUME}"
  --data-dir "${DATA_DIR}"
  --subset "${SUBSET}"
  --output-dir "${OUT_DIR}"
  --topk-pids "${TOPK_PIDS}"
  --min-samples-per-pid "${MIN_SAMPLES_PER_PID}"
  --samples-per-pid "${SAMPLES_PER_PID}"
  --selection-mode "${SELECTION_MODE}"
  --search-trials "${SEARCH_TRIALS}"
  --tsne-perplexity "${TSNE_PERPLEXITY}"
  --tsne-iters "${TSNE_ITERS}"
  --seed "${SEED}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "=========================================="
echo "特征解耦 t-SNE 可视化开始: $(date)"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${RESUME}"
echo "Subset: ${SUBSET}"
echo "Output: ${OUT_DIR}"
echo "Command: CUDA_VISIBLE_DEVICES=${GPUS} ${CMD[*]}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}" | tee "${OUT_DIR}/console.log"

echo ""
echo "完成: ${OUT_DIR}"
echo "图像: ${OUT_DIR}/tsne_disentanglement.png"
echo "表格: ${OUT_DIR}/pid_disentanglement_scores.csv, ${OUT_DIR}/selected_samples.csv"
