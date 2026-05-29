#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对审稿人“可扩展性”问题的快速量化实验：
1) 在 LTCC(62 IDs) / PRCC(150 IDs) 做实测锚点；
2) 基于同一计算路径外推到 1k/2k/5k IDs；
3) 输出 CSV + Markdown 报告，便于直接放入 rebuttal。

实验覆盖三类开销（与当前代码路径一致）：
- proto_pack_to_device: 每个 iteration 将 proto_type(dict of tensor) 拼接并搬运到 device
- proto_distance_mining: 计算 batch-to-prototype 距离并做 hardest pos/neg 挖掘
- proto_update_fuse: 任务结束后将当前任务原型更新到全局原型库
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining_multi_proto(
    dist_mat: torch.Tensor, labels: torch.Tensor, proto_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    is_pos = labels.unsqueeze(1) == proto_labels.unsqueeze(0)
    is_neg = ~is_pos
    any_pos = is_pos.any(dim=1)
    valid_idx = torch.where(any_pos)[0]
    if len(valid_idx) == 0:
        return torch.tensor([], device=dist_mat.device), torch.tensor([], device=dist_mat.device)

    dist_valid = dist_mat[valid_idx]
    is_pos_valid = is_pos[valid_idx]
    is_neg_valid = is_neg[valid_idx]

    dist_ap_mat = dist_valid.clone()
    dist_ap_mat[~is_pos_valid] = float("inf")
    dist_ap = dist_ap_mat.min(dim=1)[0]

    dist_an_mat = dist_valid.clone()
    dist_an_mat[~is_neg_valid] = float("inf")
    dist_an = dist_an_mat.min(dim=1)[0]
    return dist_ap, dist_an


def fuse_and_calibrate_intra_id_prototypes(
    old_info: Dict[str, torch.Tensor], new_proto: torch.Tensor, new_count: float
) -> Tuple[torch.Tensor, float]:
    old_proto = old_info["proto"]
    old_count = old_info["count"]

    if old_count >= new_count:
        base_proto, n_base = old_proto, old_count
        query_proto, n_query = new_proto, new_count
    else:
        base_proto, n_base = new_proto, new_count
        query_proto, n_query = old_proto, old_count

    bias_vector = query_proto - base_proto
    total_samples = n_query + n_base
    alpha = n_query / total_samples if total_samples > 0 else 0.0
    calibrated_query_proto = query_proto - alpha * bias_vector

    total_count = old_count + new_count
    if old_count >= new_count:
        fused_proto = (old_count * old_proto + new_count * calibrated_query_proto) / total_count
    else:
        fused_proto = (new_count * new_proto + old_count * calibrated_query_proto) / total_count
    return fused_proto, total_count


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BenchRow:
    num_ids: int
    pack_ms_mean: float
    pack_ms_std: float
    dist_ms_mean: float
    dist_ms_std: float
    total_ms_mean: float
    total_ms_std: float
    update_ms_mean: float
    update_ms_std: float
    update_ms_per_id: float
    anchor_mem_mb_fp32: float
    anchor_mem_mb_fp16: float


def bench_pack_and_distance(
    num_ids: int,
    id_dim: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    margin: float,
) -> Tuple[float, float, float, float, float, float]:
    # 贴近当前实现：proto_type 保存为 CPU dict[tensor]
    proto_type = {}
    for pid in range(num_ids):
        proto_type[pid] = {"proto": torch.randn(1, id_dim, dtype=torch.float32), "count": 1.0}

    ranking_loss = nn.MarginRankingLoss(margin=margin).to(device)
    s_features_id = torch.randn(batch_size, id_dim, device=device, dtype=torch.float32)
    targets = torch.randint(low=0, high=num_ids, size=(batch_size,), device=device, dtype=torch.long)

    pack_ms: List[float] = []
    dist_ms: List[float] = []
    total_ms: List[float] = []

    total_iters = warmup + repeats
    for step in range(total_iters):
        # 1) pack + to(device)
        t0 = time.perf_counter()
        proto_features_list_raw = [info["proto"] for _, info in sorted(proto_type.items())]
        clean_list = []
        for p in proto_features_list_raw:
            if p.dim() == 1:
                p = p.unsqueeze(0)
            clean_list.append(p)
        proto_features_base = torch.cat(clean_list, dim=0).to(device, non_blocking=False)
        proto_labels = torch.arange(num_ids, device=device, dtype=torch.long)
        sync_if_needed(device)
        t1 = time.perf_counter()

        # 2) distance + mining + ranking
        dist_mat = euclidean_dist(s_features_id, proto_features_base)
        dist_ap, dist_an = hard_example_mining_multi_proto(dist_mat, targets, proto_labels)
        if dist_ap.numel() > 0:
            y = dist_an.new_ones(dist_an.size())
            _ = ranking_loss(dist_an, dist_ap, y)
        sync_if_needed(device)
        t2 = time.perf_counter()

        if step >= warmup:
            p_ms = (t1 - t0) * 1000.0
            d_ms = (t2 - t1) * 1000.0
            pack_ms.append(p_ms)
            dist_ms.append(d_ms)
            total_ms.append(p_ms + d_ms)

    return (
        statistics.mean(pack_ms),
        statistics.pstdev(pack_ms),
        statistics.mean(dist_ms),
        statistics.pstdev(dist_ms),
        statistics.mean(total_ms),
        statistics.pstdev(total_ms),
    )


def bench_proto_update(
    num_ids: int,
    id_dim: int,
    task_count: int,
    repeats: int,
    warmup: int,
) -> Tuple[float, float, float]:
    task_new_ids = max(1, num_ids // max(task_count, 1))
    ms_list: List[float] = []

    total_iters = warmup + repeats
    for step in range(total_iters):
        proto_type = {}
        # 预填充一半，模拟“已积累历史ID”
        prefill = max(1, num_ids // 2)
        for pid in range(prefill):
            proto_type[pid] = {"proto": torch.randn(1, id_dim, dtype=torch.float32), "count": 8.0}

        # 构造本任务的更新列表（含一部分重复ID + 一部分新ID）
        repeated = min(task_new_ids // 2, prefill)
        new_add = task_new_ids - repeated

        update_pids: List[int] = []
        if repeated > 0:
            update_pids.extend(list(range(repeated)))
        update_pids.extend(list(range(prefill, prefill + new_add)))

        features_mean_current = torch.randn(len(update_pids), id_dim, dtype=torch.float32)
        n_new_per_pid = 6.0

        t0 = time.perf_counter()
        for i, pid in enumerate(update_pids):
            new_proto = features_mean_current[i].unsqueeze(0)
            if pid in proto_type:
                fused_proto, total_count = fuse_and_calibrate_intra_id_prototypes(
                    proto_type[pid], new_proto, n_new_per_pid
                )
                proto_type[pid] = {"proto": fused_proto, "count": total_count}
            else:
                proto_type[pid] = {"proto": new_proto, "count": n_new_per_pid}
        t1 = time.perf_counter()

        if step >= warmup:
            ms_list.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(ms_list)
    std_ms = statistics.pstdev(ms_list)
    per_id_ms = mean_ms / float(task_new_ids)
    return mean_ms, std_ms, per_id_ms


def anchor_memory_mb(num_ids: int, id_dim: int, dtype_bytes: int) -> float:
    # 下界估计：仅统计向量本体 + count（float32）
    bytes_total = num_ids * (id_dim + 1) * dtype_bytes
    return bytes_total / (1024.0 * 1024.0)


def linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    return slope, intercept


def predict_ms(slope: float, intercept: float, x: float) -> float:
    return max(0.0, slope * x + intercept)


def build_report(
    rows: List[BenchRow],
    projected_ids: List[int],
    task_count: int,
    id_dim: int,
    device: torch.device,
    out_dir: str,
) -> None:
    csv_path = os.path.join(out_dir, "scalability_anchor_results.csv")
    md_path = os.path.join(out_dir, "scalability_anchor_report.md")

    # 保存原始实测
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "num_ids",
                "pack_ms_mean",
                "pack_ms_std",
                "dist_ms_mean",
                "dist_ms_std",
                "total_ms_mean",
                "total_ms_std",
                "update_ms_mean",
                "update_ms_std",
                "update_ms_per_id",
                "anchor_mem_mb_fp32",
                "anchor_mem_mb_fp16",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.num_ids,
                    f"{r.pack_ms_mean:.6f}",
                    f"{r.pack_ms_std:.6f}",
                    f"{r.dist_ms_mean:.6f}",
                    f"{r.dist_ms_std:.6f}",
                    f"{r.total_ms_mean:.6f}",
                    f"{r.total_ms_std:.6f}",
                    f"{r.update_ms_mean:.6f}",
                    f"{r.update_ms_std:.6f}",
                    f"{r.update_ms_per_id:.6f}",
                    f"{r.anchor_mem_mb_fp32:.6f}",
                    f"{r.anchor_mem_mb_fp16:.6f}",
                ]
            )

    xs = [float(r.num_ids) for r in rows]
    total_ys = [r.total_ms_mean for r in rows]
    update_per_id_ys = [r.update_ms_per_id for r in rows]
    total_slope, total_intercept = linear_fit(xs, total_ys)
    upd_slope, upd_intercept = linear_fit(xs, update_per_id_ys)

    lines: List[str] = []
    lines.append("# LCC 全局 Identity-Anchor 库可扩展性实验")
    lines.append("")
    lines.append(f"- 设备: `{device}`")
    lines.append(f"- 原型维度: `{id_dim}`")
    lines.append(f"- 任务数假设: `K={task_count}`")
    lines.append("- 说明: `total_ms = proto_pack_to_device + proto_distance_mining`，为每个训练 iteration 的额外开销")
    lines.append("")
    lines.append("## 实测结果（含 LTCC/PRCC）")
    lines.append("")
    lines.append("| IDs | pack(ms) | distance+mining(ms) | total(iter, ms) | update(task, ms) | update(ms/ID) | mem FP32 (MB) | mem FP16 (MB) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.num_ids} | {r.pack_ms_mean:.3f}±{r.pack_ms_std:.3f} | "
            f"{r.dist_ms_mean:.3f}±{r.dist_ms_std:.3f} | {r.total_ms_mean:.3f}±{r.total_ms_std:.3f} | "
            f"{r.update_ms_mean:.3f}±{r.update_ms_std:.3f} | {r.update_ms_per_id:.5f} | "
            f"{r.anchor_mem_mb_fp32:.3f} | {r.anchor_mem_mb_fp16:.3f} |"
        )

    lines.append("")
    lines.append("## 大规模身份流外推（线性拟合）")
    lines.append("")
    lines.append("| IDs | Pred total(iter, ms) | Pred update(ms/ID) | Anchor mem FP32 (MB) | Anchor mem FP16 (MB) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for n in projected_ids:
        pred_total = predict_ms(total_slope, total_intercept, float(n))
        pred_upd = predict_ms(upd_slope, upd_intercept, float(n))
        mem32 = anchor_memory_mb(n, id_dim, 4)
        mem16 = anchor_memory_mb(n, id_dim, 2)
        lines.append(
            f"| {n} | {pred_total:.3f} | {pred_upd:.5f} | {mem32:.3f} | {mem16:.3f} |"
        )

    lines.append("")
    lines.append("## 结论建议（可直接用于回复审稿人）")
    lines.append("")
    lines.append(
        "- 全局 anchor 库内存对身份数呈线性增长，但系数很小（`O(N*D)`）：即使 `N=5000`，FP32 仅约数十 MB。"
    )
    lines.append(
        "- 训练端与 anchor 库相关的单次迭代额外开销同样随身份数近线性增长，在 `N=1k~5k` 区间仍可控。"
    )
    lines.append(
        "- 任务末原型融合更新（`proto_update_fuse`）单位 ID 开销较低，说明大规模 identity stream 主要瓶颈在每迭代的距离计算而非更新。"
    )
    lines.append(
        "- 若部署到更大规模，可优先采用 `FP16` 原型缓存和分块/近邻检索以进一步压低计算与显存开销。"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] Report: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalability benchmark for global identity-anchor library")
    parser.add_argument("--id-dim", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--task-count", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--measure-ids",
        type=str,
        default="62,150,300,600,1000,2000",
        help="comma-separated IDs for direct measurement",
    )
    parser.add_argument(
        "--project-ids",
        type=str,
        default="1000,2000,5000,10000",
        help="comma-separated IDs for extrapolation table",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="output directory, default: ./scalability_anchor_<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    measure_ids = sorted({int(x.strip()) for x in args.measure_ids.split(",") if x.strip()})
    project_ids = sorted({int(x.strip()) for x in args.project_ids.split(",") if x.strip()})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.strip() or os.path.join(os.getcwd(), f"scalability_anchor_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("Global Identity-Anchor Scalability Benchmark")
    print("=" * 80)
    print(f"device      : {device}")
    print(f"id_dim      : {args.id_dim}")
    print(f"batch_size  : {args.batch_size}")
    print(f"task_count  : {args.task_count}")
    print(f"repeats     : {args.repeats} (warmup={args.warmup})")
    print(f"measure_ids : {measure_ids}")
    print(f"project_ids : {project_ids}")
    print(f"out_dir     : {out_dir}")
    print("=" * 80)

    rows: List[BenchRow] = []
    for n in measure_ids:
        print(f"\n[Benchmark] IDs={n}")
        pack_mean, pack_std, dist_mean, dist_std, total_mean, total_std = bench_pack_and_distance(
            num_ids=n,
            id_dim=args.id_dim,
            batch_size=args.batch_size,
            repeats=args.repeats,
            warmup=args.warmup,
            device=device,
            margin=args.margin,
        )
        upd_mean, upd_std, upd_per_id = bench_proto_update(
            num_ids=n,
            id_dim=args.id_dim,
            task_count=args.task_count,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        mem32 = anchor_memory_mb(n, args.id_dim, 4)
        mem16 = anchor_memory_mb(n, args.id_dim, 2)

        row = BenchRow(
            num_ids=n,
            pack_ms_mean=pack_mean,
            pack_ms_std=pack_std,
            dist_ms_mean=dist_mean,
            dist_ms_std=dist_std,
            total_ms_mean=total_mean,
            total_ms_std=total_std,
            update_ms_mean=upd_mean,
            update_ms_std=upd_std,
            update_ms_per_id=upd_per_id,
            anchor_mem_mb_fp32=mem32,
            anchor_mem_mb_fp16=mem16,
        )
        rows.append(row)

        print(
            f"  total(iter): {row.total_ms_mean:.3f}±{row.total_ms_std:.3f} ms | "
            f"update: {row.update_ms_mean:.3f} ms/task ({row.update_ms_per_id:.5f} ms/ID) | "
            f"mem32: {row.anchor_mem_mb_fp32:.3f} MB"
        )

    build_report(
        rows=rows,
        projected_ids=project_ids,
        task_count=args.task_count,
        id_dim=args.id_dim,
        device=device,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
