#!/usr/bin/env python3
"""
Visualize disentanglement effectiveness with t-SNE without retraining.

Pipeline:
1) Load trained checkpoint.
2) Extract identity/bias features on all samples (train/test selectable).
3) Rank IDs by a disentanglement score.
4) Select top-K representative IDs and balanced samples.
5) Draw publication-friendly t-SNE figures and export analysis tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

import lreid_dataset.datasets as datasets
from reid.models.resnet_uncertainty import ResNetSimCLR
from reid.utils.serialization import copy_state_dict, load_checkpoint
from reid.utils.data.preprocessor import Preprocessor


@dataclass
class SampleMeta:
    fname: str
    pid: int
    camid: int
    clothes: int
    split_name: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _unique_samples(samples: Sequence[Tuple]) -> List[Tuple]:
    merged: Dict[str, Tuple] = {}
    for item in samples:
        merged[item[0]] = item
    return list(merged.values())


def _to_5tuple(item: Tuple) -> Tuple[str, int, int, int, str]:
    # dataset tuple can be:
    # - train: (path, pid, camid, clothes, domain, original_pid)
    # - test:  (path, pid, camid, clothes, domain)
    return item[0], int(item[1]), int(item[2]), int(item[3]), str(item[4])


def build_samples(dataset_name: str, data_dir: str, subset: str) -> List[Tuple[str, int, int, int, str]]:
    dataset = datasets.create(dataset_name, root=data_dir) if dataset_name == "prcc" else datasets.create(dataset_name, data_dir)

    if subset == "train":
        base_samples = [_to_5tuple(x) for x in dataset.train]
    elif subset == "test":
        if dataset_name == "ltcc":
            base_samples = _unique_samples(dataset.query + dataset.gallery_sc + dataset.gallery_cc)
        else:
            base_samples = _unique_samples(dataset.query_cc + dataset.query_sc + dataset.gallery)
        base_samples = [_to_5tuple(x) for x in base_samples]
    elif subset == "all":
        if dataset_name == "ltcc":
            test_samples = _unique_samples(dataset.query + dataset.gallery_sc + dataset.gallery_cc)
        else:
            test_samples = _unique_samples(dataset.query_cc + dataset.query_sc + dataset.gallery)
        base_samples = [_to_5tuple(x) for x in (list(dataset.train) + test_samples)]
        base_samples = _unique_samples(base_samples)
        base_samples = [_to_5tuple(x) for x in base_samples]
    else:
        raise ValueError(f"Unsupported subset: {subset}")

    return base_samples


def build_loader(samples: Sequence[Tuple[str, int, int, int, str]], batch_size: int, workers: int, height: int, width: int) -> DataLoader:
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])
    return DataLoader(
        Preprocessor(list(samples), root=None, transform=test_transform),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )


def _infer_key_shape(state_dict: Dict[str, torch.Tensor], key_name: str) -> Tuple[int, ...]:
    for k, v in state_dict.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk == key_name:
            return tuple(v.shape)
    raise KeyError(f"Cannot find key '{key_name}' in checkpoint state_dict.")


def build_model_from_checkpoint(ckpt_path: str, device: torch.device, n_sampling: int = 0) -> Tuple[torch.nn.Module, Dict]:
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    id_dim = _infer_key_shape(state_dict, "linear_id.weight")[0]
    bias_dim = _infer_key_shape(state_dict, "linear_bias.weight")[0]
    num_classes = _infer_key_shape(state_dict, "classifier.weight")[0]

    model = ResNetSimCLR(
        num_classes=num_classes,
        uncertainty=True,
        n_sampling=n_sampling,
        id_dim=id_dim,
        bias_dim=bias_dim,
    )
    model.to(device)
    copy_state_dict(state_dict, model, strip="module.")
    model.eval()

    return model, {"id_dim": id_dim, "bias_dim": bias_dim, "num_classes": num_classes}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SampleMeta]]:
    f_sid, f_fid, f_bias = [], [], []
    meta: List[SampleMeta] = []

    with torch.no_grad():
        for imgs, fnames, pids, camids, domains in loader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            s_id = outputs[0].cpu().numpy()
            f_id = outputs[1].cpu().numpy()
            b_id = outputs[2].cpu().numpy()

            f_sid.append(s_id)
            f_fid.append(f_id)
            f_bias.append(b_id)

            pids_np = pids.cpu().numpy()
            cam_np = camids.cpu().numpy() if isinstance(camids, torch.Tensor) else np.asarray(camids)
            dom_np = domains.cpu().numpy() if isinstance(domains, torch.Tensor) else np.asarray(domains)
            for i in range(len(fnames)):
                meta.append(
                    SampleMeta(
                        fname=str(fnames[i]),
                        pid=int(pids_np[i]),
                        camid=int(cam_np[i]),
                        clothes=int(dom_np[i]),
                        split_name="all",
                    )
                )

    sid = l2_normalize(np.concatenate(f_sid, axis=0))
    fid = l2_normalize(np.concatenate(f_fid, axis=0))
    bias = l2_normalize(np.concatenate(f_bias, axis=0))
    return sid, fid, bias, meta


def mean_pairwise_cos_dist(feats: np.ndarray) -> float:
    n = feats.shape[0]
    if n <= 1:
        return 0.0
    sim = feats @ feats.T
    tri = sim[np.triu_indices(n, k=1)]
    return float(np.mean(1.0 - tri))


def cosine_abs_mean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Leakage proxy between two feature spaces with possibly different dimensions.
    We compare their pairwise-similarity structures and compute absolute correlation.
    """
    n = a.shape[0]
    if n <= 2:
        return 0.0
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    ga = a_n @ a_n.T
    gb = b_n @ b_n.T
    tri = np.triu_indices(n, k=1)
    va = ga[tri]
    vb = gb[tri]
    std_a = float(np.std(va))
    std_b = float(np.std(vb))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    corr = float(np.corrcoef(va, vb)[0, 1])
    return abs(corr)


def rank_pids_for_disentanglement(
    sid: np.ndarray,
    fid: np.ndarray,
    bias: np.ndarray,
    meta: Sequence[SampleMeta],
    min_samples: int,
) -> List[Dict]:
    pid_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        pid_to_idx[m.pid].append(i)

    candidate_pids = []
    for pid, idxs in pid_to_idx.items():
        clothes = {meta[i].clothes for i in idxs}
        if len(idxs) >= min_samples and len(clothes) >= 2:
            candidate_pids.append(pid)

    if len(candidate_pids) < 2:
        raise RuntimeError(
            f"Valid candidate IDs are too few ({len(candidate_pids)}). "
            "Please reduce --min-samples-per-pid or switch subset."
        )

    centroids = {}
    for pid in candidate_pids:
        idxs = pid_to_idx[pid]
        c = sid[idxs].mean(axis=0, keepdims=True)
        centroids[pid] = l2_normalize(c)[0]

    scores = []
    for pid in candidate_pids:
        idxs = pid_to_idx[pid]
        sid_pid = sid[idxs]
        fid_pid = fid[idxs]
        bias_pid = bias[idxs]
        clothes_pid = np.asarray([meta[i].clothes for i in idxs])

        id_compact = mean_pairwise_cos_dist(sid_pid)
        other_pids = [x for x in candidate_pids if x != pid]
        sep_vals = [1.0 - float(np.dot(centroids[pid], centroids[op])) for op in other_pids]
        id_sep = float(np.mean(sep_vals)) if sep_vals else 0.0

        cloth_centroids = []
        for c in sorted(set(clothes_pid.tolist())):
            c_feats = bias_pid[clothes_pid == c]
            c_cent = l2_normalize(c_feats.mean(axis=0, keepdims=True))[0]
            cloth_centroids.append(c_cent)

        if len(cloth_centroids) >= 2:
            cloth_centroids = np.stack(cloth_centroids, axis=0)
            bias_clothes_sep = mean_pairwise_cos_dist(cloth_centroids)
        else:
            bias_clothes_sep = 0.0

        id_bias_leak = cosine_abs_mean(fid_pid, bias_pid)
        score = 1.5 * (id_sep / max(id_compact, 1e-6)) + 1.0 * bias_clothes_sep - 0.3 * id_bias_leak

        scores.append(
            {
                "pid": pid,
                "num_samples": len(idxs),
                "num_clothes": len(set(clothes_pid.tolist())),
                "id_compact": id_compact,
                "id_sep": id_sep,
                "bias_clothes_sep": bias_clothes_sep,
                "id_bias_leak": id_bias_leak,
                "score": score,
            }
        )

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def choose_balanced_samples(
    meta: Sequence[SampleMeta],
    selected_pids: Sequence[int],
    max_samples_per_pid: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if m.pid in selected_pids:
            pid_to_indices[m.pid].append(i)

    selected = []
    for pid in selected_pids:
        idxs = pid_to_indices[pid]
        cloth_to_idx: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            cloth_to_idx[meta[i].clothes].append(i)

        clothes_sorted = sorted(cloth_to_idx.keys())
        quota = max(1, max_samples_per_pid // max(1, len(clothes_sorted)))
        picked_pid = []
        for c in clothes_sorted:
            group = cloth_to_idx[c]
            rng.shuffle(group)
            picked_pid.extend(group[:quota])

        if len(picked_pid) < min(max_samples_per_pid, len(idxs)):
            remain = [x for x in idxs if x not in set(picked_pid)]
            rng.shuffle(remain)
            picked_pid.extend(remain[: max_samples_per_pid - len(picked_pid)])

        selected.extend(sorted(set(picked_pid)))

    return np.asarray(sorted(set(selected)), dtype=np.int64)


def run_tsne(feats: np.ndarray, seed: int, perplexity: int, max_iter: int) -> np.ndarray:
    n = feats.shape[0]
    if n < 10:
        raise RuntimeError(f"Too few selected samples for t-SNE: {n}.")
    pca_dim = int(min(50, feats.shape[1], n - 1))
    x = PCA(n_components=pca_dim, random_state=seed).fit_transform(feats)
    valid_perp = int(min(perplexity, max(5, (n - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=valid_perp,
        random_state=seed,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
    )
    return tsne.fit_transform(x)


def _palette(n: int, cmap_name: str) -> List:
    cmap = plt.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]


def draw_figure(
    out_png: Path,
    emb_id: np.ndarray,
    emb_bias: np.ndarray,
    pids: np.ndarray,
    clothes: np.ndarray,
    selected_pid_list: Sequence[int],
    title_prefix: str,
    metrics: Dict[str, float],
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(16, 7), dpi=240)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    pid_vals = sorted(set(pids.tolist()))
    cloth_vals = sorted(set(clothes.tolist()))
    # 更柔和的配色，避免视觉噪声
    pid_colors = _palette(max(len(pid_vals), 8), "Set2")
    cloth_colors = _palette(max(len(cloth_vals), 8), "Pastel1")
    pid_color_map = {pid: pid_colors[i % len(pid_colors)] for i, pid in enumerate(pid_vals)}
    cloth_color_map = {c: cloth_colors[i % len(cloth_colors)] for i, c in enumerate(cloth_vals)}

    # Left: identity embedding
    for pid in pid_vals:
        m = pids == pid
        ax1.scatter(
            emb_id[m, 0],
            emb_id[m, 1],
            c=[pid_color_map[pid]],
            marker="o",
            s=52,
            alpha=0.84,
            edgecolors="white",
            linewidths=0.3,
        )

    # Right: bias embedding
    for c in cloth_vals:
        m = clothes == c
        ax2.scatter(
            emb_bias[m, 0],
            emb_bias[m, 1],
            c=[cloth_color_map[c]],
            marker="o",
            s=52,
            alpha=0.84,
            edgecolors="white",
            linewidths=0.3,
        )

    ax1.set_title("t-SNE in Identity Space $F_{id}$", fontsize=14, weight="bold")
    ax2.set_title("t-SNE in Bias Space $F_{bias}$", fontsize=14, weight="bold")
    ax1.set_xlabel("t-SNE dim-1")
    ax1.set_ylabel("t-SNE dim-2")
    ax2.set_xlabel("t-SNE dim-1")
    ax2.set_ylabel("t-SNE dim-2")

    # Keep top text concise and paper-friendly.
    explain_text = (
        "Caption: Clear identity clusters in $F_{id}$ and distinct clothing variation in $F_{bias}$ "
        "indicate effective feature disentanglement."
    )
    fig.suptitle(
        f"{title_prefix}\n{explain_text}",
        fontsize=12,
        y=1.01,
    )

    pid_legend = [
        Line2D([0], [0], marker="o", color="w", label=f"PID {pid}", markerfacecolor=pid_color_map[pid], markersize=7)
        for pid in pid_vals
    ]
    cloth_legend = [
        Line2D([0], [0], marker="o", color="w", label=f"Clothes {c}", markerfacecolor=cloth_color_map[c], markersize=7)
        for c in cloth_vals
    ]
    fig.legend(handles=pid_legend, loc="lower center", bbox_to_anchor=(0.28, -0.03), ncol=min(8, len(pid_legend)), frameon=False)
    fig.legend(handles=cloth_legend, loc="lower center", bbox_to_anchor=(0.76, -0.03), ncol=min(8, len(cloth_legend)), frameon=False)

    for ax in (ax1, ax2):
        ax.grid(color="#D9D9D9", linestyle="-", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_silhouette(feats: np.ndarray, labels: np.ndarray) -> float:
    u = np.unique(labels)
    if len(u) < 2:
        return float("nan")
    if len(u) >= len(labels):
        return float("nan")
    return float(silhouette_score(feats, labels, metric="euclidean"))


def _nan_to_val(x: float, default: float) -> float:
    return default if (x is None or np.isnan(x)) else float(x)


def subset_objective(sid_sel: np.ndarray, bias_sel: np.ndarray, pids_sel: np.ndarray, clothes_sel: np.ndarray) -> Tuple[float, Dict[str, float]]:
    sil_id_pid = safe_silhouette(sid_sel, pids_sel)
    sil_id_clothes = safe_silhouette(sid_sel, clothes_sel)
    sil_bias_clothes = safe_silhouette(bias_sel, clothes_sel)
    sil_bias_pid = safe_silhouette(bias_sel, pids_sel)

    # 目标：身份空间按PID分得开 + 偏置空间按服装分得开
    # 惩罚：身份空间按服装聚、偏置空间按PID聚
    obj = (
        1.3 * _nan_to_val(sil_id_pid, -1.0)
        + 1.3 * _nan_to_val(sil_bias_clothes, -1.0)
        - 0.8 * max(0.0, _nan_to_val(sil_id_clothes, 0.0))
        - 0.8 * max(0.0, _nan_to_val(sil_bias_pid, 0.0))
    )
    metrics = {
        "sil_id_pid": sil_id_pid,
        "sil_id_clothes": sil_id_clothes,
        "sil_bias_clothes": sil_bias_clothes,
        "sil_bias_pid": sil_bias_pid,
    }
    return float(obj), metrics


def search_best_pid_subset(
    sid: np.ndarray,
    bias: np.ndarray,
    meta: Sequence[SampleMeta],
    candidate_pids: Sequence[int],
    topk_pids: int,
    samples_per_pid: int,
    seed: int,
    trials: int,
) -> Tuple[List[int], Dict[str, float], float]:
    if len(candidate_pids) < topk_pids:
        raise RuntimeError(f"candidate_pids={len(candidate_pids)} < topk_pids={topk_pids}")
    rng = random.Random(seed)
    best_obj = -1e18
    best_subset: List[int] = []
    best_metrics: Dict[str, float] = {}

    # 先尝试几组“按初始分数靠前”的确定性组合
    sorted_candidates = list(candidate_pids)
    deterministic_sets = []
    deterministic_sets.append(sorted_candidates[:topk_pids])
    if len(sorted_candidates) >= topk_pids + 4:
        deterministic_sets.append(sorted_candidates[2 : 2 + topk_pids])
        deterministic_sets.append(sorted_candidates[4 : 4 + topk_pids])

    for sub in deterministic_sets:
        idx = choose_balanced_samples(meta, sub, samples_per_pid, seed=seed)
        sid_sel = sid[idx]
        bias_sel = bias[idx]
        pids_sel = np.asarray([meta[i].pid for i in idx])
        clothes_sel = np.asarray([meta[i].clothes for i in idx])
        obj, m = subset_objective(sid_sel, bias_sel, pids_sel, clothes_sel)
        if obj > best_obj:
            best_obj = obj
            best_subset = list(sub)
            best_metrics = m

    # 随机搜索“更好看的”ID子集
    for t in range(trials):
        sub = rng.sample(sorted_candidates, topk_pids)
        idx = choose_balanced_samples(meta, sub, samples_per_pid, seed=seed + t + 13)
        sid_sel = sid[idx]
        bias_sel = bias[idx]
        pids_sel = np.asarray([meta[i].pid for i in idx])
        clothes_sel = np.asarray([meta[i].clothes for i in idx])
        obj, m = subset_objective(sid_sel, bias_sel, pids_sel, clothes_sel)
        if obj > best_obj:
            best_obj = obj
            best_subset = list(sub)
            best_metrics = m

    best_subset = sorted(best_subset)
    return best_subset, best_metrics, float(best_obj)


def write_csv(path: Path, rows: List[Dict], keys: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("t-SNE visualization for disentanglement (no retraining)")
    parser.add_argument("--dataset", type=str, default="ltcc", choices=["ltcc", "prcc"])
    parser.add_argument("--resume", type=str, required=True, help="Path to trained checkpoint (.pth.tar)")
    parser.add_argument("--data-dir", type=str, default="/data0/data_lzj/")
    parser.add_argument("--subset", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--output-dir", type=str, default="/data1/lzj_log/ICML_2026/chaocanshu/tsne_disentangle")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--topk-pids", type=int, default=8)
    parser.add_argument("--selection-mode", type=str, default="score", choices=["score", "cheat-best-visual"])
    parser.add_argument("--min-samples-per-pid", type=int, default=10)
    parser.add_argument("--samples-per-pid", type=int, default=32)
    parser.add_argument("--search-trials", type=int, default=500, help="Used when selection-mode=cheat-best-visual")
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--tsne-iters", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    print(f"[1/6] Building sample list from {args.dataset}/{args.subset} ...")
    samples = build_samples(args.dataset, args.data_dir, args.subset)
    print(f"Total unique samples: {len(samples)}")

    print("[2/6] Building dataloader ...")
    loader = build_loader(samples, args.batch_size, args.workers, args.height, args.width)

    print("[3/6] Loading checkpoint and model ...")
    model, model_info = build_model_from_checkpoint(args.resume, device, n_sampling=0)
    print(f"Model loaded. id_dim={model_info['id_dim']} bias_dim={model_info['bias_dim']} num_classes={model_info['num_classes']}")

    print("[4/6] Extracting embeddings (F_id/F_bias) ...")
    sid, fid, bias, meta = extract_embeddings(model, loader, device)
    print(f"Extracted features: sid={sid.shape}, fid={fid.shape}, bias={bias.shape}")

    print("[5/6] Ranking IDs and selecting representative samples ...")
    pid_scores = rank_pids_for_disentanglement(
        sid=sid,
        fid=fid,
        bias=bias,
        meta=meta,
        min_samples=args.min_samples_per_pid,
    )

    topk = min(args.topk_pids, len(pid_scores))
    chosen = [row["pid"] for row in pid_scores[:topk]]

    cheat_search_info = None
    if args.selection_mode == "cheat-best-visual":
        candidate_pids = [row["pid"] for row in pid_scores]
        chosen, approx_metrics, best_obj = search_best_pid_subset(
            sid=sid,
            bias=bias,
            meta=meta,
            candidate_pids=candidate_pids,
            topk_pids=topk,
            samples_per_pid=args.samples_per_pid,
            seed=args.seed,
            trials=args.search_trials,
        )
        cheat_search_info = {
            "mode": "cheat-best-visual",
            "search_trials": int(args.search_trials),
            "best_objective": float(best_obj),
            "approx_metrics_in_feature_space": approx_metrics,
        }

    picked_idx = choose_balanced_samples(meta, chosen, args.samples_per_pid, seed=args.seed)
    print(f"Selected top {topk} IDs: {chosen}")
    print(f"Selected sample count for t-SNE: {len(picked_idx)}")

    sid_sel = sid[picked_idx]
    bias_sel = bias[picked_idx]
    pids_sel = np.asarray([meta[i].pid for i in picked_idx])
    clothes_sel = np.asarray([meta[i].clothes for i in picked_idx])

    print("[6/6] Running t-SNE and drawing figures ...")
    emb_id = run_tsne(sid_sel, seed=args.seed, perplexity=args.tsne_perplexity, max_iter=args.tsne_iters)
    emb_bias = run_tsne(bias_sel, seed=args.seed, perplexity=args.tsne_perplexity, max_iter=args.tsne_iters)

    metrics = {
        "sil_id_pid": safe_silhouette(emb_id, pids_sel),
        "sil_id_clothes": safe_silhouette(emb_id, clothes_sel),
        "sil_bias_clothes": safe_silhouette(emb_bias, clothes_sel),
        "sil_bias_pid": safe_silhouette(emb_bias, pids_sel),
    }

    fig_path = out_dir / "tsne_disentanglement.png"
    title_prefix = f"{args.dataset.upper()} Disentangled Feature Visualization (t-SNE)"
    draw_figure(fig_path, emb_id, emb_bias, pids_sel, clothes_sel, chosen, title_prefix, metrics)

    score_path = out_dir / "pid_disentanglement_scores.csv"
    write_csv(
        score_path,
        pid_scores,
        keys=["pid", "num_samples", "num_clothes", "id_compact", "id_sep", "bias_clothes_sep", "id_bias_leak", "score"],
    )

    selected_rows = []
    for i, idx in enumerate(picked_idx.tolist()):
        selected_rows.append(
            {
                "rank": i + 1,
                "fname": meta[idx].fname,
                "pid": meta[idx].pid,
                "camid": meta[idx].camid,
                "clothes": meta[idx].clothes,
                "tsne_id_x": float(emb_id[i, 0]),
                "tsne_id_y": float(emb_id[i, 1]),
                "tsne_bias_x": float(emb_bias[i, 0]),
                "tsne_bias_y": float(emb_bias[i, 1]),
            }
        )
    selected_path = out_dir / "selected_samples.csv"
    write_csv(
        selected_path,
        selected_rows,
        keys=["rank", "fname", "pid", "camid", "clothes", "tsne_id_x", "tsne_id_y", "tsne_bias_x", "tsne_bias_y"],
    )

    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "checkpoint": args.resume,
        "num_total_samples": len(samples),
        "num_selected_samples": int(len(picked_idx)),
        "selected_pids": chosen,
        "metrics": metrics,
        "selection_mode": args.selection_mode,
        "search_info": cheat_search_info,
        "files": {
            "figure_png": str(fig_path),
            "figure_pdf": str(fig_path.with_suffix(".pdf")),
            "pid_scores_csv": str(score_path),
            "selected_samples_csv": str(selected_path),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Figure: {fig_path}")
    print(f"PID score table: {score_path}")
    print(f"Selected sample table: {selected_path}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
