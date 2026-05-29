#!/bin/bash
# LTCC 任务顺序方差实验（fangcha）
# 说明：
# - 使用“已有基线顺序结果” + “两个新 task-order-seed 顺序结果”
# - 自动输出每个顺序结果 + 均值 + 方差（便于论文整理）

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
GPUS="0"
LOG_ROOT="/data1/lzj_log/ICML_2026/chaocanshu"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DRY_RUN="0"

# 默认把当前顺序视为 seed=1（你现有代码常用）
BASELINE_ORDER_SEED="1"
ORDER_SEEDS=("11" "23")   # 两个新的顺序
BASELINE_LOG_RES=""        # 传入已有基线 log_res.txt 路径

EXTRA_ARGS=()

print_help() {
  cat << EOF
用法:
  bash run_ltcc_fangcha.sh [选项] [-- 透传给 continual_train.py 的参数]

选项:
  --gpus <ids>               指定可见 GPU（默认: ${GPUS}）
  --python <bin>             Python 命令（默认: ${PYTHON_BIN}）
  --log-root <path>          日志根目录（默认: ${LOG_ROOT}）
  --tag <name>               本次运行标记（默认: 时间戳）
  --baseline-seed <int>      当前顺序对应 seed（默认: ${BASELINE_ORDER_SEED}）
  --seed-a <int>             新顺序1 seed（默认: ${ORDER_SEEDS[0]}）
  --seed-b <int>             新顺序2 seed（默认: ${ORDER_SEEDS[1]}）
  --baseline-log-res <path>  已有基线的 log_res.txt（推荐提供）
  --dry-run                  仅打印命令，不实际训练
  -h, --help                 显示帮助

示例:
  bash run_ltcc_fangcha.sh --gpus 1 --baseline-log-res /path/to/log_res.txt
  bash run_ltcc_fangcha.sh --gpus 0 --seed-a 7 --seed-b 19 -- --epochs0 40 --epochs 30
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
    --baseline-seed)
      BASELINE_ORDER_SEED="$2"; shift 2 ;;
    --seed-a)
      ORDER_SEEDS[0]="$2"; shift 2 ;;
    --seed-b)
      ORDER_SEEDS[1]="$2"; shift 2 ;;
    --baseline-log-res)
      BASELINE_LOG_RES="$2"; shift 2 ;;
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

RUN_BASE="${LOG_ROOT%/}/ltcc_fangcha_${RUN_TAG}"
MASTER_LOG="${RUN_BASE}/ltcc_fangcha.log"
REPORT_MD="${RUN_BASE}/ltcc_fangcha_report.md"
CONFIG_TSV="${RUN_BASE}/order_runs.tsv"

mkdir -p "${RUN_BASE}"

{
  echo "=========================================="
  echo "LTCC 任务顺序方差实验开始: $(date)"
  echo "项目目录: ${PROJECT_DIR}"
  echo "日志目录: ${RUN_BASE}"
  echo "GPU: ${GPUS}"
  echo "基线顺序 seed: ${BASELINE_ORDER_SEED}"
  echo "新增顺序 seeds: ${ORDER_SEEDS[*]}"
  echo "已有基线 log_res: ${BASELINE_LOG_RES:-<未提供>}"
  echo "=========================================="
} | tee -a "${MASTER_LOG}"

echo -e "run_name\torder_seed\trun_dir\tlog_res_path\tstatus" > "${CONFIG_TSV}"

# 先登记基线（可能来自已有结果，不重跑）
if [[ -n "${BASELINE_LOG_RES}" ]]; then
  echo -e "baseline\t${BASELINE_ORDER_SEED}\t-\t${BASELINE_LOG_RES}\texisting" >> "${CONFIG_TSV}"
else
  echo "[提示] 未提供 --baseline-log-res，将在本脚本内补跑 baseline。" | tee -a "${MASTER_LOG}"
fi

run_single_case() {
  local run_name="$1"
  local order_seed="$2"
  local run_dir="${RUN_BASE}/${run_name}_seed${order_seed}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/continual_train.py"
    --dataset ltcc
    --logs-dir "${run_dir}"
    --task-order-seed "${order_seed}"
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "" | tee -a "${MASTER_LOG}"
  echo ">>> 开始: ${run_name}, task-order-seed=${order_seed}" | tee -a "${MASTER_LOG}"
  echo ">>> 运行目录: ${run_dir}" | tee -a "${MASTER_LOG}"
  echo ">>> 命令: CUDA_VISIBLE_DEVICES=${GPUS} ${cmd[*]}" | tee -a "${MASTER_LOG}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "${run_name}\t${order_seed}\t${run_dir}\t${run_dir}/log_res.txt\tdry-run" >> "${CONFIG_TSV}"
    echo "[DRY-RUN] 跳过实际训练" | tee -a "${MASTER_LOG}"
    return 0
  fi

  if CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}" 2>&1 | tee "${run_dir}/console.log" >> "${MASTER_LOG}"; then
    echo -e "${run_name}\t${order_seed}\t${run_dir}\t${run_dir}/log_res.txt\tsuccess" >> "${CONFIG_TSV}"
    echo ">>> 状态: 完成" | tee -a "${MASTER_LOG}"
  else
    echo -e "${run_name}\t${order_seed}\t${run_dir}\t${run_dir}/log_res.txt\tfailed" >> "${CONFIG_TSV}"
    echo ">>> 状态: 失败（已继续后续实验）" | tee -a "${MASTER_LOG}"
  fi
}

# 如果没有外部基线，则补跑 baseline
if [[ -z "${BASELINE_LOG_RES}" ]]; then
  run_single_case "baseline" "${BASELINE_ORDER_SEED}"
fi

# 跑两个新顺序
run_single_case "order_a" "${ORDER_SEEDS[0]}"
run_single_case "order_b" "${ORDER_SEEDS[1]}"

echo "" | tee -a "${MASTER_LOG}"
echo ">>> 开始汇总均值/方差表格..." | tee -a "${MASTER_LOG}"

"${PYTHON_BIN}" - << 'PY' "${CONFIG_TSV}" "${REPORT_MD}" "${MASTER_LOG}"
import csv
import os
import re
import statistics
import sys

config_tsv, report_md, master_log = sys.argv[1], sys.argv[2], sys.argv[3]

def parse_ltcc_metrics(log_res_path):
    if not log_res_path or log_res_path == "-" or not os.path.exists(log_res_path):
        return None
    text = open(log_res_path, "r", encoding="utf-8", errors="ignore").read()
    sc = re.findall(r"^SC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    cc = re.findall(r"^CC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    if not sc or not cc:
        return None
    sc_map, sc_r1 = map(float, sc[-1])
    cc_map, cc_r1 = map(float, cc[-1])
    return {
        "sc_map": sc_map, "sc_r1": sc_r1,
        "cc_map": cc_map, "cc_r1": cc_r1
    }

rows = []
with open(config_tsv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

ordered_names = ["baseline", "order_a", "order_b"]
row_map = {r["run_name"]: r for r in rows}

valid_metrics = []
lines = []
lines.append("# LTCC 任务顺序方差实验结果")
lines.append("")
lines.append("| 运行 | task-order-seed | SC mAP | SC R1 | CC mAP | CC R1 | 状态 |")
lines.append("|---|---:|---:|---:|---:|---:|---|")

for name in ordered_names:
    r = row_map.get(name)
    if r is None:
        lines.append(f"| {name} | - | - | - | - | - | 未执行 |")
        continue
    m = parse_ltcc_metrics(r["log_res_path"])
    if m is None:
        lines.append(f"| {name} | {r['order_seed']} | - | - | - | - | {r['status']} |")
    else:
        valid_metrics.append(m)
        lines.append(
            f"| {name} | {r['order_seed']} | "
            f"{m['sc_map']:.1f} | {m['sc_r1']:.1f} | {m['cc_map']:.1f} | {m['cc_r1']:.1f} | {r['status']} |"
        )

def mean_var(values):
    if len(values) == 0:
        return None, None
    mean_v = sum(values) / len(values)
    var_v = sum((x - mean_v) ** 2 for x in values) / len(values)  # 总体方差
    return mean_v, var_v

sc_map_list = [m["sc_map"] for m in valid_metrics]
sc_r1_list = [m["sc_r1"] for m in valid_metrics]
cc_map_list = [m["cc_map"] for m in valid_metrics]
cc_r1_list = [m["cc_r1"] for m in valid_metrics]

sc_map_mean, sc_map_var = mean_var(sc_map_list)
sc_r1_mean, sc_r1_var = mean_var(sc_r1_list)
cc_map_mean, cc_map_var = mean_var(cc_map_list)
cc_r1_mean, cc_r1_var = mean_var(cc_r1_list)

lines.append("")
lines.append("| 统计项 | SC mAP | SC R1 | CC mAP | CC R1 |")
lines.append("|---|---:|---:|---:|---:|")

if sc_map_mean is not None:
    lines.append(f"| Mean | {sc_map_mean:.2f} | {sc_r1_mean:.2f} | {cc_map_mean:.2f} | {cc_r1_mean:.2f} |")
    lines.append(f"| Variance | {sc_map_var:.4f} | {sc_r1_var:.4f} | {cc_map_var:.4f} | {cc_r1_var:.4f} |")
else:
    lines.append("| Mean | - | - | - | - |")
    lines.append("| Variance | - | - | - | - |")

md = "\n".join(lines) + "\n"
os.makedirs(os.path.dirname(report_md), exist_ok=True)
with open(report_md, "w", encoding="utf-8") as f:
    f.write(md)

with open(master_log, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("LTCC 方差实验汇总表\n")
    f.write("=" * 80 + "\n")
    f.write(md)

print(md)
PY

echo "" | tee -a "${MASTER_LOG}"
echo "实验全部结束: $(date)" | tee -a "${MASTER_LOG}"
echo "总日志: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "汇总表: ${REPORT_MD}" | tee -a "${MASTER_LOG}"

