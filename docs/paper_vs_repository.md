# Paper vs Repository — DAE

> PDFs received 2026-09-08 (`medianomaly.pdf`, 25 pp; `DAE paper.pdf`, 15 pp).
> "DAE paper/source" column below is now filled from the actual PDFs. Benchmark
> reference numbers come from `medianomaly.pdf` (same protocol we ran);
> `DAE paper.pdf` values are cited only as excluded context (different setting).

Upstream commit: `507201cb8e604b7d970c388784f55c06bb32d013`.

| Component | DAE paper/source | MedIAnomaly implementation (verified in code) | Final reproduction |
|---|---|---|---|
| Noise resolution | `DAE paper.pdf` Table 3: DAE (α = 16, σ = 0.2) | `utils/dae_worker.py: self.noise_res = 16` | `16` — match |
| Noise std | `DAE paper.pdf` Table 3: DAE (α = 16, σ = 0.2) | `utils/dae_worker.py: self.noise_std = 0.2`, `torch.normal(mean=0, std=0.2)` at 16×16 | `0.2` — match |
| Noise upsampling | Not specified (client PDF pending) | `F.interpolate(..., size=input_size, mode='bilinear', align_corners=True)` | bilinear to `input_size` |
| Noise translation | Not specified (client PDF pending) | `torch.roll` by random `roll_x, roll_y` in `[0, input_size)` | random roll |
| Foreground mask | Not specified (client PDF pending; brief states brain/BraTS mask) | `if dataset in ['brain','brats']: ns *= (x > x.min())` | applied for `brats` |
| Noise post-transform | — | `ns = (ns - 0.5) * 2; noisy = x + ns` | as implemented, unchanged |
| Reconstruction target | Healthy/clean image (per brief) | `criterion(img_clean, net(noisy))` in `DAEWorker.train_epoch` | clean image |
| Input size | Not specified (client PDF pending) | default `64`; `train_eval.sh` DAE loop uses `128` | `128×128` |
| Architecture | Not specified (client PDF pending; brief says customized U-Net) | `networks/unet.py::UNet(depth=5, wf=6, ReflectionPad+WNConv2d+CustomSwish+GroupNorm, avg_pool down, ConvTranspose2d up)`; `BaseWorker` builds `UNet(in_c, in_c)` for `dae` | unchanged U-Net |
| Input/output channels (BraTS) | Not specified | `in_c=1` (default map), single-channel `L` images | `1→1` |
| Loss | Not specified (client PDF pending) | `losses.py::AELoss` = mean squared error `(x - x_hat)^2`; anomaly map = channel-mean SE | MSE (`AELoss`), unchanged |
| Dataset | BraTS2021 per client slide/brief | `BraTSAD` expects `~/MedIAnomaly-Data/BraTS2021/{train,test/{normal,tumor,annotation}}` | BraTS2021 real data only |
| Training epochs | Not specified | `options.py: brats→250` | `250` unless feasibility forces documented reduction |
| Batch size | Not specified | default `64`; DAE loop `-bs 16` | `16` |
| Optimizer/LR | Not specified | `Adam(lr=1e-3, wd=0)`, no scheduler (commented out) | unchanged |
| Metrics | `medianomaly.pdf` p.9: image AUC + AP; pixel APpix + ⌈Dice⌉ (best Dice at optimal test operating point); pixel AUC explicitly NOT used | `ae_worker.evaluate`: image `AUC/AP`; if `brats`: `PixAUC/PixAP/BestDice/BestThresh` + mean normal/abnormal scores | AUC/AP + PixAP/PixAUC/Dice; accuracy only supplemental. Note: our PixAUC has no published counterpart (paper excludes it by design) |
| Seed | Paper: mean±std over 3 seeds | `--train-seed None` → random `1..999999` if unset | seed 0 (single-seed run vs 3-seed published mean — stated, not hidden) |

## Published benchmark results vs reproduced (DAE + BraTS2021)

Source: `medianomaly.pdf` Table 6 p.11 (DAE row, BraTS2021: AUC 85.9±1.0, AP 93.4±0.5)
and Table 7 p.12 (DAE row: APpix 75.5±0.7, ⌈Dice⌉ 71.1±0.6). Paper terms mapped:
AP→AP, APpix→PixAP, ⌈Dice⌉→BestDice.

| Metric | Published (mean±std) | Reproduced (seed 0) | Diff | |Diff| | % | Within 1 std |
|---|---|---|---|---|---|---|
| AUC | 0.859±0.010 | 0.85699 | −0.00201 | 0.00201 | 0.23% | yes |
| AP | 0.934±0.005 | 0.93286 | −0.00114 | 0.00114 | 0.12% | yes |
| PixAP | 0.755±0.007 | 0.74949 | −0.00551 | 0.00551 | 0.73% | yes |
| Dice | 0.711±0.006 | 0.70583 | −0.00517 | 0.00517 | 0.73% | yes |
| PixAUC | not reported (excluded, p.9) | 0.96751 | N/A | N/A | N/A | — |

Excluded context (different experimental setting, NOT compared): `DAE paper.pdf`
p.9 Table 3, DAE(α=16,σ=0.2) on that paper's own BraTS pipeline — pixel AUPRC
0.833±0.005, dDice 0.773±0.004. Machine-readable: `artifacts/results/reference.json`.

## Known paper↔repo differences to confirm once PDFs arrive

1. Generic DAE literature often uses Gaussian or simplex noise at full resolution; MedIAnomaly uses **coarse 16×16 → bilinear → roll → foreground-masked** corruption. Keep repo behavior.
2. Standard U-Net uses BatchNorm+ReLU+maxpool; MedIAnomaly uses **GroupNorm+CustomSwish+WNConv+avg_pool+ReflectionPad**. Do not “fix” to textbook U-Net.
3. `train_eval.sh` `gpu=7` is a hardware assumption, not a method parameter — adapt device, keep method params.
