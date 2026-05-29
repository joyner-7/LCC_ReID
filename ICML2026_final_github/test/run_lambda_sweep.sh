#!/bin/bash
# 一键运行 lambda 消融实验：
# 固定其中两个 lambda，扫描第三个 lambda（每个扫描 3 个值，默认值仅记录不运行）
# 同时覆盖 LTCC 与 PRCC，并在日志中输出汇总表格。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认值（用于本次消融基线）
DEFAULT_RECON_WEIGHT="1.2"   # λ1 (Lrec)
DEFAULT_AF_WEIGHT="1.5"      # λ2 (Lalign)
DEFAULT_FID_KD_WEIGHT="0.5"  # λ3 (Lisd)

# 扫描值（不包含默认值）
RECON_SWEEP_VALUES=("0.8" "1.6" "2.4")
AF_SWEEP_VALUES=("0.5" "1.0" "2.0")
FID_KD_SWEEP_VALUES=("0.2" "1.0" "2.0")

GPUS="0"
PYTHON_BIN="python"
LOG_ROOT="/data1/lzj_log/ICML_2026/chaocanshu"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DRY_RUN="0"

EXTRA_ARGS=()

NUM_DATASETS=2
NUM_SCAN_PER_LAMBDA=3
NUM_LAMBDAS=3
NUM_TRAIN_CASES=$((NUM_DATASETS * NUM_SCAN_PER_LAMBDA * NUM_LAMBDAS))
NUM_DEFAULT_RECORDS=$((NUM_DATASETS * NUM_LAMBDAS))
NUM_TOTAL_ROWS=$((NUM_TRAIN_CASES + NUM_DEFAULT_RECORDS))

print_help() {
  cat << EOF
用法:
  bash run_lambda_sweep.sh [选项] [-- 传给 continual_train.py 的其他参数]

选项:
  --gpus <ids>          指定可见 GPU，例如 "0" 或 "0,1"（默认: ${GPUS}）
  --python <bin>        Python 命令（默认: ${PYTHON_BIN}）
  --log-root <path>     总日志根目录（默认: ${LOG_ROOT}）
  --tag <name>          本次运行标记（默认: 时间戳）
  --dry-run             仅打印将执行的命令，不实际训练
  -h, --help            显示帮助

示例:
  bash run_lambda_sweep.sh --gpus 1
  bash run_lambda_sweep.sh --gpus 0,1 -- --epochs0 40 --epochs 30 --middle_test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --log-root)
      LOG_ROOT="$2"; shift 2 ;;
    --tag)
      RUN_TAG="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN="1"; shift ;;
    -h|--help)
      print_help; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    *)
      echo "[错误] 未知参数: $1"
      print_help
      exit 1 ;;
  esac
done

RUN_BASE="${LOG_ROOT%/}/lambda_sweep_${RUN_TAG}"
MASTER_LOG="${RUN_BASE}/lambda_sweep_master.log"
CONFIG_TSV="${RUN_BASE}/runs_config.tsv"
REPORT_MD="${RUN_BASE}/lambda_sweep_report.md"

mkdir -p "${RUN_BASE}"

echo "==========================================" | tee -a "${MASTER_LOG}"
echo "Lambda 消融实验开始: $(date)" | tee -a "${MASTER_LOG}"
echo "项目目录: ${PROJECT_DIR}" | tee -a "${MASTER_LOG}"
echo "日志目录: ${RUN_BASE}" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPUS}" | tee -a "${MASTER_LOG}"
echo "总损失: Ltotal = Lreid + λ1*Lrec + λ2*Lalign + λ3*Lisd" | tee -a "${MASTER_LOG}"
echo "实验数量: 实际训练 ${NUM_TRAIN_CASES} 个, 默认值记录 ${NUM_DEFAULT_RECORDS} 条, 汇总总条目 ${NUM_TOTAL_ROWS} 条" | tee -a "${MASTER_LOG}"
echo "==========================================" | tee -a "${MASTER_LOG}"

{
  echo -e "dataset\tablation\trecon_weight\taf_weight\tfid_kd_weight\trun_flag\trun_dir"
} > "${CONFIG_TSV}"

append_row() {
  local dataset="$1"
  local ablation="$2"
  local rw="$3"
  local aw="$4"
  local kw="$5"
  local run_flag="$6"
  local run_dir="$7"
  echo -e "${dataset}\t${ablation}\t${rw}\t${aw}\t${kw}\t${run_flag}\t${run_dir}" >> "${CONFIG_TSV}"
}

run_single_case() {
  local dataset="$1"
  local ablation="$2"
  local rw="$3"
  local aw="$4"
  local kw="$5"

  local case_name="${ablation}_rw${rw}_aw${aw}_kw${kw}"
  local run_dir="${RUN_BASE}/${dataset}/${case_name}"
  mkdir -p "${run_dir}"

  append_row "${dataset}" "${ablation}" "${rw}" "${aw}" "${kw}" "1" "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/continual_train.py"
    --dataset "${dataset}"
    --logs-dir "${run_dir}"
    --recon-weight "${rw}"
    --AF_weight "${aw}"
    --fid-kd-weight "${kw}"
  )

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "" | tee -a "${MASTER_LOG}"
  echo ">>> 开始: dataset=${dataset}, ablation=${ablation}, rw=${rw}, aw=${aw}, kw=${kw}" | tee -a "${MASTER_LOG}"
  echo ">>> 运行目录: ${run_dir}" | tee -a "${MASTER_LOG}"
  echo ">>> 命令: CUDA_VISIBLE_DEVICES=${GPUS} ${cmd[*]}" | tee -a "${MASTER_LOG}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] 跳过实际训练" | tee -a "${MASTER_LOG}"
    return 0
  fi

  if CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}" 2>&1 | tee "${run_dir}/console.log" >> "${MASTER_LOG}"; then
    echo ">>> 状态: 完成" | tee -a "${MASTER_LOG}"
  else
    echo ">>> 状态: 失败（已继续后续实验）" | tee -a "${MASTER_LOG}"
  fi
}

for dataset in ltcc prcc; do
  # 记录默认值（不运行）
  append_row "${dataset}" "recon_weight(default_only)" "${DEFAULT_RECON_WEIGHT}" "${DEFAULT_AF_WEIGHT}" "${DEFAULT_FID_KD_WEIGHT}" "0" "-"
  append_row "${dataset}" "AF_weight(default_only)" "${DEFAULT_RECON_WEIGHT}" "${DEFAULT_AF_WEIGHT}" "${DEFAULT_FID_KD_WEIGHT}" "0" "-"
  append_row "${dataset}" "fid_kd_weight(default_only)" "${DEFAULT_RECON_WEIGHT}" "${DEFAULT_AF_WEIGHT}" "${DEFAULT_FID_KD_WEIGHT}" "0" "-"

  # λ1: recon_weight 扫描，固定 λ2/λ3 为默认
  for rw in "${RECON_SWEEP_VALUES[@]}"; do
    run_single_case "${dataset}" "recon_weight" "${rw}" "${DEFAULT_AF_WEIGHT}" "${DEFAULT_FID_KD_WEIGHT}"
  done

  # λ2: AF_weight 扫描，固定 λ1/λ3 为默认
  for aw in "${AF_SWEEP_VALUES[@]}"; do
    run_single_case "${dataset}" "AF_weight" "${DEFAULT_RECON_WEIGHT}" "${aw}" "${DEFAULT_FID_KD_WEIGHT}"
  done

  # λ3: fid_kd_weight 扫描，固定 λ1/λ2 为默认
  for kw in "${FID_KD_SWEEP_VALUES[@]}"; do
    run_single_case "${dataset}" "fid_kd_weight" "${DEFAULT_RECON_WEIGHT}" "${DEFAULT_AF_WEIGHT}" "${kw}"
  done
done

echo "" | tee -a "${MASTER_LOG}"
echo ">>> 所有训练已完成，开始汇总表格..." | tee -a "${MASTER_LOG}"

"${PYTHON_BIN}" "${PROJECT_DIR}/summarize_lambda_sweep.py" \
  --config "${CONFIG_TSV}" \
  --output-md "${REPORT_MD}" \
  --append-log "${MASTER_LOG}" | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "实验全部结束: $(date)" | tee -a "${MASTER_LOG}"
echo "总日志: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "汇总表: ${REPORT_MD}" | tee -a "${MASTER_LOG}"

