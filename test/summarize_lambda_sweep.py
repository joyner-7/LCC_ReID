#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, Optional, Tuple, List


def parse_metrics(dataset: str, run_dir: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    log_res = os.path.join(run_dir, "log_res.txt")
    if not os.path.exists(log_res):
        return None, None, None, None

    with open(log_res, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if dataset == "ltcc":
        sc_matches = re.findall(r"^SC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
        cc_matches = re.findall(r"^CC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    elif dataset == "prcc":
        sc_matches = re.findall(r"^PRCC_SC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
        cc_matches = re.findall(r"^PRCC_CC:\t([\d.]+)\t([\d.]+)", text, flags=re.M)
    else:
        return None, None, None, None

    sc_map, sc_r1 = (float(sc_matches[-1][0]), float(sc_matches[-1][1])) if sc_matches else (None, None)
    cc_map, cc_r1 = (float(cc_matches[-1][0]), float(cc_matches[-1][1])) if cc_matches else (None, None)
    return sc_map, sc_r1, cc_map, cc_r1


def fmt_num(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def build_markdown(rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# Lambda 消融实验汇总")
    lines.append("")
    lines.append("| 数据集 | 变动项 | recon_weight(λ1) | AF_weight(λ2) | fid_kd_weight(λ3) | SC mAP | SC R1 | CC mAP | CC R1 | 状态 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        dataset = row["dataset"]
        ablation = row["ablation"]
        rw = row["recon_weight"]
        aw = row["af_weight"]
        kw = row["fid_kd_weight"]
        run_flag = row["run_flag"]
        run_dir = row["run_dir"]

        if run_flag == "0":
            status = "已记录(默认值未运行)"
            sc_map = sc_r1 = cc_map = cc_r1 = None
        else:
            sc_map, sc_r1, cc_map, cc_r1 = parse_metrics(dataset, run_dir)
            status = "完成" if (sc_map is not None and cc_map is not None) else "未完成/失败"

        lines.append(
            f"| {dataset} | {ablation} | {rw} | {aw} | {kw} | "
            f"{fmt_num(sc_map)} | {fmt_num(sc_r1)} | {fmt_num(cc_map)} | {fmt_num(cc_r1)} | {status} |"
        )

    lines.append("")
    lines.append("说明：SC/CC 指对应数据集协议下的 same-clothes / cross-clothes（或标准/换装）评估。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 lambda 消融实验结果为 Markdown 表格")
    parser.add_argument("--config", required=True, help="run 配置 tsv 文件路径")
    parser.add_argument("--output-md", required=True, help="输出 markdown 路径")
    parser.add_argument("--append-log", default="", help="可选：附加写入到总日志文件")
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.config, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    # 保持可读性：按数据集、变动项、是否运行排序
    rows.sort(key=lambda x: (x["dataset"], x["ablation"], int(x["run_flag"]), float(x["recon_weight"]), float(x["af_weight"]), float(x["fid_kd_weight"])))

    md = build_markdown(rows)

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    if args.append_log:
        with open(args.append_log, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Lambda 消融实验表格汇总\n")
            f.write("=" * 80 + "\n")
            f.write(md)

    print(md)


if __name__ == "__main__":
    main()
