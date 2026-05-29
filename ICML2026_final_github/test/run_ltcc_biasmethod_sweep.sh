#!/bin/bash
# LTCC bias-swapping 采样策略对比实验
# 方法：random / hard / semi-hard
# 输出目录位于 /data1/lzj_log/ICML_2026/chaocanshu，自动带时间戳避免覆盖。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
GPUS="0"
LOG_ROOT="/data1/lzj_log/ICML_2026/chaocanshu"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DRY_RUN="0"

METHODS=("random" "hard" "semi-hard")
SEMI_LOW="0.3"
SEMI_HIGH="0.6"

EXTRA_ARGS=()

print_help() {
  cat << EOF
用法:
  bash run_ltcc_biasmethod_sweep.sh [选项] [-- 传给 continual_train.py 的其他参数]

选项:
  --gpus <ids>          指定可见 GPU，例如 "0" 或 "1"（默认: ${GPUS}）
  --python <bin>        Python 命令（默认: ${PYTHON_BIN}）
  --log-root <path>     日志根目录（默认: ${LOG_ROOT}）
  --tag <name>          本次运行标记（默认: 时间戳）
  --semi-low <ratio>    semi-hard 区间下界（默认: ${SEMI_LOW}）
  --semi-high <ratio>   semi-hard 区间上界（默认: ${SEMI_HIGH}）
  --dry-run             仅打印将执行命令，不实际训练
  -h, --help            显示帮助

示例:
  bash run_ltcc_biasmethod_sweep.sh --gpus 1
  bash run_ltcc_biasmethod_sweep.sh --gpus 0 -- --epochs0 40 --epochs 30 --middle_test
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
    --semi-low)
      SEMI_LOW="$2"; shift 2 ;;
    --semi-high)
      SEMI_HIGH="$2"; shift 2 ;;
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

RUN_BASE="${LOG_ROOT%/}/ltcc_biasmethod_${RUN_TAG}"
MASTER_LOG="${RUN_BASE}/ltcc_biasmethod.log"
REPORT_MD="${RUN_BASE}/ltcc_biasmethod_report.md"
CONFIG_TSV="${RUN_BASE}/biasmethod_runs.tsv"

mkdir -p "${RUN_BASE}"

{
  echo "=========================================="
  echo "LTCC biasmethod 对比实验开始: $(date)"
  echo "项目目录: ${PROJECT_DIR}"
  echo "日志目录: ${RUN_BASE}"
  echo "GPU: ${GPUS}"
  echo "方法列表: ${METHODS[*]}"
  echo "semi-hard 区间: [${SEMI_LOW}, ${SEMI_HIGH}]"
  echo "=========================================="
} | tee -a "${MASTER_LOG}"

echo -e "method\trun_dir\tstatus" > "${CONFIG_TSV}"

run_single_case() {
  local method="$1"
  local run_dir="${RUN_BASE}/method_${method}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/continual_train.py"
    --dataset ltcc
    --logs-dir "${run_dir}"
    --bias-swap-method "${method}"
    --bias-swap-semihard-low "${SEMI_LOW}"
    --bias-swap-semihard-high "${SEMI_HIGH}"
  )

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "" | tee -a "${MASTER_LOG}"
  echo ">>> 开始: LTCC bias-swap-method=${method}" | tee -a "${MASTER_LOG}"
  echo ">>> 运行目录: ${run_dir}" | tee -a "${MASTER_LOG}"
  echo ">>> 命令: CUDA_VISIBLE_DEVICES=${GPUS} ${cmd[*]}" | tee -a "${MASTER_LOG}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "${method}\t${run_dir}\tdry-run" >> "${CONFIG_TSV}"
    echo "[DRY-RUN] 跳过实际训练" | tee -a "${MASTER_LOG}"
    return 0
  fi

  if CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}" 2>&1 | tee "${run_dir}/console.log" >> "${MASTER_LOG}"; then
    echo -e "${method}\t${run_dir}\tsuccess" >> "${CONFIG_TSV}"
    echo ">>> 状态: 完成" | tee -a "${MASTER_LOG}"
  else
    echo -e "${method}\t${run_dir}\tfailed" >> "${CONFIG_TSV}"
    echo ">>> 状态: 失败（已继续后续实验）" | tee -a "${MASTER_LOG}"
  fi
}

for method in "${METHODS[@]}"; do
  run_single_case "${method}"
done

echo "" | tee -a "${MASTER_LOG}"
echo ">>> 开始汇总 LTCC biasmethod 结果..." | tee -a "${MASTER_LOG}"

"${PYTHON_BIN}" - << 'PY' "${CONFIG_TSV}" "${REPORT_MD}" "${MASTER_LOG}"
import csv
import os
import re
import sys

config_tsv, report_md, master_log = sys.argv[1], sys.argv[2], sys.argv[3]

rows = []
with open(config_tsv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

order = {"random": 0, "hard": 1, "semi-hard": 2}
rows.sort(key=lambda x: order.get(x["method"], 99))

def parse_ltcc_metrics(run_dir: str):
    log_res = os.path.join(run_dir, "log_res.txt")
    if not os.path.exists(log_res):
        return None, None, None, None
    text = open(log_res, "r", encoding="utf-8", errors="ignore").read()
    sc = re.findall(r"^SC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    cc = re.findall(r"^CC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    if not sc or not cc:
        return None, None, None, None
    sc_map, sc_r1 = map(float, sc[-1])
    cc_map, cc_r1 = map(float, cc[-1])
    return sc_map, sc_r1, cc_map, cc_r1

def fmt(v):
    return "-" if v is None else f"{v:.1f}"

lines = []
lines.append("# LTCC bias-swapping 采样策略对比")
lines.append("")
lines.append("| bias-swap method | SC mAP | SC R1 | CC mAP | CC R1 | 状态 |")
lines.append("|---|---:|---:|---:|---:|---|")

for r in rows:
    method = r["method"]
    run_dir = r["run_dir"]
    status = r["status"]
    sc_map, sc_r1, cc_map, cc_r1 = parse_ltcc_metrics(run_dir)
    if status != "success":
        show_status = status
    else:
        show_status = "完成" if sc_map is not None else "完成(未解析到指标)"
    lines.append(
        f"| {method} | {fmt(sc_map)} | {fmt(sc_r1)} | {fmt(cc_map)} | {fmt(cc_r1)} | {show_status} |"
    )

md = "\n".join(lines) + "\n"
os.makedirs(os.path.dirname(report_md), exist_ok=True)
with open(report_md, "w", encoding="utf-8") as f:
    f.write(md)

with open(master_log, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("LTCC biasmethod 汇总表\n")
    f.write("=" * 80 + "\n")
    f.write(md)

print(md)
PY

echo "" | tee -a "${MASTER_LOG}"
echo "实验全部结束: $(date)" | tee -a "${MASTER_LOG}"
echo "总日志: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "汇总表: ${REPORT_MD}" | tee -a "${MASTER_LOG}"

