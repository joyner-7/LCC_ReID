# Lifelong Clothes-Changing Person Re-Identification

Official implementation of our lifelong (continual) **clothes-changing person re-identification (CC-ReID)** framework.
The model learns from a stream of incremental tasks while resisting catastrophic forgetting, by disentangling
identity-relevant features from clothing/appearance bias and maintaining a global identity-anchor prototype library.

## Highlights

- **Feature disentanglement** into an identity space (`f_id`) and a bias/appearance space (`f_bias`) via an
  encoder–decoder with self- and cross-reconstruction (bias swapping).
- **Anti-forgetting prototype alignment**: a global identity-anchor library is continually fused and
  bias-calibrated across tasks, and used as a margin-ranking constraint during training.
- **Identity-stable distillation** (`f_id` KD) on repeated identities between the old and new model.
- **Dynamic classifier expansion**: the classifier grows on demand as new identities appear.
- **Old/new model fusion** after each task to balance stability and plasticity.

## Repository Structure

```
.
├── continual_train.py          # Main entry point (training + evaluation loop)
├── reid/                       # Core ReID library
│   ├── models/                 # ResNet backbone with uncertainty / disentanglement heads
│   ├── loss/                   # Triplet, cross-entropy, reconstruction, uncertainty losses
│   ├── trainer.py              # Per-task training logic
│   ├── evaluators.py           # Feature extraction + CMC / mAP evaluation
│   ├── evaluation_metrics/     # Ranking metrics (CMC, mAP)
│   └── utils/                  # Data loaders, samplers, transforms, schedulers, IO
├── lreid_dataset/              # Incremental dataset builders (LTCC, PRCC)
│   └── datasets/get_data_loaders.py
├── tools/                      # Result logging helpers
├── test/                       # Reproducibility scripts (hyper-parameter sweeps, t-SNE, etc.)
├── scalability_anchor_*/       # Prototype-library scalability study reports
├── requirement.txt             # Python dependencies
└── LICENSE
```

## Installation

```bash
# Python 3.8+ and a CUDA-capable GPU are recommended
conda create -n cc_reid python=3.8 -y
conda activate cc_reid

# Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch torchvision

# Install the remaining dependencies
pip install -r requirement.txt
```

## Datasets

Two clothes-changing benchmarks are supported. Set `--data-dir` to the root that contains the following layout:

```
<data-dir>/
├── LTCC_ReID/
│   ├── info/        # cloth-(un)change_id_(train|test).txt
│   ├── train/
│   └── test/
└── prcc/
    └── rgb/
        ├── train/
        └── test/
```

- **LTCC**: [Long-Term Cloth-Changing](https://naiq.github.io/LTCC_Perosn_ReID.html)
- **PRCC**: [Person Re-id under moderate Clothing Change](https://www.isee-ai.cn/~yangqize/clothing.html)

## Usage

### Training

```bash
# PRCC (incremental tasks split by ID-clothing combinations)
CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --dataset prcc \
    --data-dir /path/to/data \
    --logs-dir logs/prcc_main

# LTCC
CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --dataset ltcc \
    --data-dir /path/to/data \
    --logs-dir logs/ltcc_main \
    --middle_test
```

Multi-GPU training is supported through `DataParallel`, e.g. `CUDA_VISIBLE_DEVICES=0,1`.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `prcc` | Benchmark to use (`ltcc` or `prcc`) |
| `--epochs0` / `--epochs` | `40` / `30` | Epochs for the first / subsequent tasks |
| `--id-dim` / `--bias-dim` | `1536` / `512` | Identity / bias feature dimensions |
| `--recon-weight` | `1.2` | Reconstruction loss weight |
| `--AF_weight` | `1.5` | Anti-forgetting (prototype alignment) weight |
| `--fid-kd-weight` | `0.5` | `f_id` knowledge-distillation weight |
| `--triplet-margin` | `0.3` | Triplet loss margin |
| `--bias-swap-method` | `random` | Bias swapping strategy (`random`/`hard`/`semi-hard`) |
| `--middle_test` | off | Run evaluation during training |

Run `python continual_train.py --help` for the full list.

### Evaluation Protocol

- **LTCC**: Standard / General (SC) and Cloth-Changing (CC) settings.
- **PRCC**: Same-Clothes (query B vs. gallery A) and Cross-Clothes (query C vs. gallery A) settings.

Results (mAP / Rank-1) are printed to stdout and written to `log_res.txt` inside `--logs-dir`.

### Reproducibility Scripts

The `test/` directory contains helper scripts for the ablations and analyses reported in the paper
(hyper-parameter sweeps, bias-swap strategy comparison, t-SNE disentanglement visualization, and the
prototype-library scalability study). They wrap `continual_train.py`; adjust the paths inside before running.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
