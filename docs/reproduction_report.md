# Reproduction Report — DAE on BraTS2021

## 1. Objective
Reproduce MedIAnomaly DAE (reconstruction-based, healthy-only training, coarse-noise
denoising, U-Net, reconstruction-error scoring) on real BraTS2021 for classification +
segmentation. Status at this commit: pipeline validated on real data; full 250-epoch
P100 run in progress (kernel `arkosarkarhehe/dae-brats-p100-train`, RUNNING). No
numbers fabricated.

## 2. Source repositories
- Upstream: `https://github.com/caiyu6666/MedIAnomaly` (`upstream/main`).
- Delivery: `https://github.com/Anamitra-Sarkar/DAE-method`, branch `client/dae-reproduction`
  (based on upstream) + `main` (delivery subset).

## 3. Exact Git commit
- Upstream base: `507201cb8e604b7d970c388784f55c06bb32d013`.
- Delivery commits: see `git log` on `main` / `client/dae-reproduction`.

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

## 12. Results
Pending kernel output. Placeholders not filled. Raw `metrics.txt` + `final_results.*`
will be pulled to `artifacts/` on completion.

## 13. Reference comparison
Pending (no benchmark table in env; client PDFs outstanding). Will compute
reproduced-reference deltas per metric on arrival.

## 14. Visual results
`artifacts/visualizations/dae_debug.png` (DEBUG ONLY: clean/coarse/upsampled/
masked/noisy/recon, seed 0). Full recon/residual montages pending eval `-save`.

## 15. Runtime
Pending. Training started on P100; CPU tiny-run ~12+ min for partial epoch.

## 16. Deviations
1. Device: `-g 0` (P100) / CPU shim (Codespace) instead of repo `gpu=7`; math unchanged.
2. Torch on P100 2.4.1 cu118 for sm_60 (vs README 2.1.2); CPU env keeps 2.1.2.
3. `wandb` offline. 4. Epoch override via harness (`options.py` ignores `--train-epochs`).
No noise/arch/loss/dataset substitution.

## 17. Limitations
CPU-only dev host; client PDFs absent (paper column TBD); 250ep needs GPU;
single fold.

## 18. Reproduction conclusion
PARTIAL (pipeline reproduced on real data; numerical reproduction pending P100).
Outcome class will be updated to Exact/Consistent/Partial/Failed once metrics land.
