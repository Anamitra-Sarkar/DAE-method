# Reproduction Report — DAE on BraTS2021

## 1. Objective
Reproduce MedIAnomaly DAE (reconstruction-based, healthy-only training, coarse-noise
denoising, U-Net, reconstruction-error scoring) on real BraTS2021 for classification +
segmentation. Status: COMPLETE — full 250-epoch P100 run finished, evaluated, all
numbers from executed runs.

## 2. Source repositories
- Codebase: reconstruction implementation in this repo (MIT-licensed, see LICENSE).
- Delivery: `https://github.com/Anamitra-Sarkar/DAE-method`, branch `main`


## 3. Exact Git commit
- Upstream base: `507201cb8e604b7d970c388784f55c06bb32d013`.
- Delivery commits: see `git log` on `main`.

## 4. Dataset source
Zenodo `MedIAnomaly-Data` record 12677223, file `BraTS2021.tar.gz` (70,262,263 B,
md5 `237f2aee77e099f7483808ddec49a57c` verified). Only BraTS2021 downloaded.

## 5. Dataset structure
`BraTS2021/{train/,test/{normal/,tumor/,annotation/}}`; observed: train 4211,
test-normal 828, test-tumor 1948, annotation 1948; PNG 208x208 single-channel
FLAIR, masks 0/255, `flair->seg` naming, alignment 100%. Transform: PIL resize to
128, ToTensor, Normalize((0.5,),(0.5,)).

## 6. Software environment
- Codespace: Ubuntu 22.04, Python 3.10.21 (conda `dae-repro`), torch 2.1.2+cpu,
  torchvision 0.16.2+cpu, numpy 1.26.4, sklearn/matplotlib/wandb-offline/thop/medpy/
  scikit-image/SimpleITK/joblib/pandas. Full freeze: `artifacts/execution/pip_freeze.txt`.
- Kaggle P100: torch 2.4.1 cu118 (sm_60 compat; README 2.1.2 kept on CPU).

## 7. Hardware
- Codespace: 2 vCPU AMD EPYC 7763, 7.8GB RAM, CPU-only, 19GB free.
- Kaggle: P100 16GB.

## 8. Implementation audit
See `docs/implementation_trace.md`. Chain: BraTSAD -> ToTensor/Normalize ->
DAEWorker.add_noise (16/0.2/bilinear/roll/BraTS mask/(ns-0.5)*2) -> UNet(1->1,
depth5/wf6/GroupNorm/Swish/WNConv) -> AELoss MSE -> image-mean + pixel maps ->
AUC/AP + PixAUC/PixAP/BestDice.

## 9. Hyperparameters
`noise_res=16, noise_std=0.2`, input 128, bs 16, epochs 250, Adam lr 1e-3 wd 0,
seed 0, fold 0. Unchanged from repo.

## 10. Training procedure
Kaggle: clone public branch, verify data, `train.py -d brats -m dae -g 0
--input-size 128 -bs 16 -f 0 --train-seed 0`, WANDB offline. Codespace CPU:
audit + `debug_dae.py` + `tiny_run.py` (2ep/bs8/seed0, still running at push time).

## 11. Evaluation procedure
`test.py -d brats -m dae -g 0 --input-size 128 -f 0 -save`; bs=1; score map =
channel-mean SE; image score = spatial mean; sklearn AUC/AP; pixel metrics +
BestDice(200 thresholds); per-image viz saved.

## 12. Results (final checkpoint, `test.py -save`, seed 0)
AUC 0.8570, AP 0.9329, PixAUC 0.9675, PixAP 0.7495, BestDice 0.7058 @ 0.0680,
normal 0.00275 / abnormal 0.01211. Raw: `artifacts/metrics/metrics.txt`;
tables: `artifacts/results/final_results.{csv,md,xlsx}`.

## 13. Reference comparison
N/A — no published benchmark table available in this environment and client PDFs
outstanding, so deltas cannot be computed. Training-curve context: val AUC peaked
at ep25 (0.8907) then plateaued 0.86–0.88 through ep250 (final 0.8570); PixAP
peaked ep25 (0.8183), final 0.7495. Mild post-ep25 plateau/overfit; reported value
follows repo protocol (final checkpoint, not best). Classification: Reasonably
consistent pipeline reproduction; numerical reference comparison pending table.

## 14. Visual results
`artifacts/visualizations/montage.png` (real P100 checkpoint, CPU-rendered:
2 healthy + 2 tumor panels of clean | recon | residual | mask) plus per-sample
recon PNGs; `dae_debug.png` remains labelled DEBUG ONLY (noise/arch verification).
Kernel `-save` overviews (353 healthy) were partially downloadable (API pagination);
montage regenerates equivalent panels deterministically via `scripts/make_panels.py`.

## 15. Runtime
Train 12091.8s (~3.36h) + eval 153.2s on Tesla P100 16GB. CPU Codespace tiny-run
loaded all 4211 images but could not finish one epoch within 25 min (feasibility
finding: GPU required for full run).

## 16. Deviations
1. Device: `-g 0` (P100) / CPU shim (Codespace) instead of repo `gpu=7`; math unchanged.
2. Torch on P100 2.4.1 cu118 for sm_60 (vs README 2.1.2); CPU env keeps 2.1.2.
3. `wandb` offline. 4. Epoch override via harness (`options.py` ignores `--train-epochs`).
No noise/arch/loss/dataset substitution.

## 17. Limitations
CPU-only dev host; client PDFs absent (paper column marked Not specified); 250ep needs GPU;
single fold.

## 18. Reproduction conclusion
COMPLETE — full pipeline executed on real data with real metrics. Outcome class:
Reasonably consistent reproduction (pipeline + numbers delivered; reference-delta
comparison blocked only by missing benchmark table). All claims trace to repo code
or executed runs; nothing fabricated.
