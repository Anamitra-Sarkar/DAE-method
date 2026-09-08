# Paper vs Repository — DAE

> Client PDFs (`DAE paper.pdf`, `medianomaly.pdf`, `DAE_Method_Explained.md`, `DAE_Implementation_Guide.md`) were **not available in this environment at audit time** (2026-09-08). Values below for “DAE paper/source” are therefore marked `Not specified (client PDF pending)` where the repo is the only verified source. The task brief itself states the verified DAE facts; those are used as the interim source and flagged accordingly. This file must be updated once PDFs are supplied — never invent missing source values.

Upstream commit: `507201cb8e604b7d970c388784f55c06bb32d013`.

| Component | DAE paper/source | MedIAnomaly implementation (verified in code) | Final reproduction |
|---|---|---|---|
| Noise resolution | Not specified (client PDF pending; brief states coarse noise) | `utils/dae_worker.py: self.noise_res = 16` | `16` |
| Noise std | Not specified (client PDF pending; brief states 0.2) | `utils/dae_worker.py: self.noise_std = 0.2`, `torch.normal(mean=0, std=0.2)` at 16×16 | `0.2` |
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
| Metrics | Anomaly AUC/AP + segmentation where applicable (per brief) | `ae_worker.evaluate`: image `AUC/AP`; if `brats`: `PixAUC/PixAP/BestDice/BestThresh` + mean normal/abnormal scores | AUC/AP + PixAP/PixAUC/Dice; accuracy only supplemental |
| Seed | Not specified | `--train-seed None` → random `1..999999` if unset | fixed seed recorded in manifest |

## Known paper↔repo differences to confirm once PDFs arrive

1. Generic DAE literature often uses Gaussian or simplex noise at full resolution; MedIAnomaly uses **coarse 16×16 → bilinear → roll → foreground-masked** corruption. Keep repo behavior.
2. Standard U-Net uses BatchNorm+ReLU+maxpool; MedIAnomaly uses **GroupNorm+CustomSwish+WNConv+avg_pool+ReflectionPad**. Do not “fix” to textbook U-Net.
3. `train_eval.sh` `gpu=7` is a hardware assumption, not a method parameter — adapt device, keep method params.
