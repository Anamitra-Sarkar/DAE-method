# DAE Reproduction — BraTS2021

## Objective
Reproduce the MedIAnomaly DAE method on real BraTS2021 brain-MRI data:
normal-only training, coarse-noise denoising with a customized U-Net,
image-level detection + pixel-level localization.

## What was reproduced
- Method: DAE (`reconstruction/utils/dae_worker.py`: `noise_res=16`, `noise_std=0.2`,
  bilinear upsample to input size, random roll, BraTS foreground mask, clean target).
- Dataset: real BraTS2021 (train 4211 normal; test 828 normal + 1948 tumor + 1948 masks).
- Model: `UNet(in=1,out=1,depth=5,wf=6)` ~31.04M params; loss `AELoss` (MSE).
- Metrics: image AUC/AP; pixel PixAUC/PixAP/BestDice (BraTS only). Accuracy supplemental only.
- Codebase: DAE reconstruction implementation (MIT-licensed, see LICENSE), source revision `507201c`; branch `main`.

## Environment
- Audit/dev: Codespace (2 cores, 8GB RAM, CPU-only, Python 3.10 conda env `dae-repro`, torch 2.1.2+cpu, numpy 1.26.4).
- Full training: Kaggle P100 GPU kernel `arkosarkarhehe/dae-brats-p100-train`
  (torch 2.4.1 cu118 for sm_60 compat; method unchanged). CPU-only Codespace cannot
  train 31M UNet x 4211 images in reasonable time (CPU tiny-run proved data
  loading; full training required the GPU).

## Installation
```bash
conda create -y -n dae-repro python=3.10 && conda activate dae-repro
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib wandb thop medpy scikit-image SimpleITK joblib pandas scipy openpyxl
pip install 'numpy<2'   # torch 2.1.2 needs numpy 1.x
export WANDB_MODE=offline
```

## Dataset setup
```bash
mkdir -p ~/MedIAnomaly-Data && cd ~/MedIAnomaly-Data
wget -O BraTS2021.tar.gz https://zenodo.org/api/records/12677223/files/BraTS2021.tar.gz/content
echo "237f2aee77e099f7483808ddec49a57c  BraTS2021.tar.gz" | md5sum -c -
tar -xzf BraTS2021.tar.gz   # -> BraTS2021/{train,test/{normal,tumor,annotation}}
python scripts/inspect_brats.py
```

## Training (GPU; repo's `gpu=7` does not exist — use real id)
```bash
cd reconstruction
python train.py -d brats -m dae -g 0 --input-size 128 -bs 16 -f 0 --train-seed 0
```

## Evaluation
```bash
python test.py -d brats -m dae -g 0 --input-size 128 -f 0 -save
```
Outputs: `~/Experiment/MedIAnomaly/brats/dae/fold_0/{train_options.txt,metrics.txt,checkpoints/model.pt,test_results/vis/...}`.
CPU-only hosts: `scripts/cpu_shim.py` patches `.cuda()`/load device only (no method change);
GPU hosts run unmodified code. Never run `./train_eval.sh` (full 7-dataset benchmark).

## Results (final, Kaggle P100, seed 0, 250 epochs)
`test.py` on the final checkpoint (`artifacts/metrics/metrics.txt`):

| Metric | Published (`medianomaly.pdf`) | Reproduced | Diff | Within 1 std |
|---|---|---|---|---|
| AUC | 0.859±0.010 (Table 6, p.11) | 0.8570 | −0.00201 (0.23%) | yes |
| AP | 0.934±0.005 (Table 6, p.11) | 0.9329 | −0.00114 (0.12%) | yes |
| PixAP | 0.755±0.007 (Table 7, p.12, APpix) | 0.7495 | −0.00551 (0.73%) | yes |
| Dice | 0.711±0.006 (Table 7, p.12, ⌈Dice⌉) | 0.7058 (thresh 0.0680) | −0.00517 (0.73%) | yes |
| PixAUC | not reported (excluded by paper, p.9) | 0.9675 | N/A | — |
| normal/abnormal score | — | 0.00275 / 0.01211 | — | — |

Paper = mean±std over 3 seeds; reproduction = single seed 0. **Near-exact/
consistent reproduction: every metric within one published std.**
Training curve (val): ep1 AUC 0.786 → ep25 peak 0.891 → plateau ~0.86–0.88
(ep50–250); final 0.857. Best-observed ep25; reported value is the protocol's
final checkpoint, not the best. Full table + per-metric sources:
`artifacts/results/final_results.{csv,md,xlsx}`, machine-readable
`artifacts/results/reference.json`. Runtime: train
12091.8s, eval 153.2s on Tesla P100. Checkpoint: release
`v0.1-dae-brats-p100` (sha256 `d862a224…05505e`), pointer in
`artifacts/checkpoints/README.md`. Visuals: `artifacts/visualizations/montage.png`
(real checkpoint, 2 healthy + 2 tumor).

## Hardware
- Codespace: AMD EPYC 7763 (2 vCPU), 7.8GB RAM, 19GB workspace free, no GPU.
- Kaggle: P100 16GB (sm_60), torch 2.4.1 cu118.

## Limitations
- Reference values now filled from `medianomaly.pdf` (Tables 6–7); `DAE paper.pdf`
  values cited as excluded context only (different setting).
- Full 250-epoch run only feasible on GPU; CPU Codespace limited to audit/smoke/tiny-run.
- `options.py` ignores `--train-epochs` (uses dataset map); epoch override done via harness.

## File map
- `reconstruction/` — reconstruction implementation (unmodified method).
- `docs/{implementation_trace,paper_vs_repository,experiment_scope,DAE_methodology,reproduction_report}.md`
- `scripts/{inspect_brats,debug_dae,cpu_shim,tiny_run,verify_reproduction,run_all_checks,run_dae_reproduction}`
- `artifacts/{dataset_manifest.json,reproduction_manifest.json,model_summary.txt,execution/,metrics/,results/,visualizations/,checkpoints/}`
