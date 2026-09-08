# DAE Implementation Trace — MedIAnomaly (`reconstruction/`)

Upstream: `caiyu6666/MedIAnomaly`, commit `507201cb8e604b7d970c388784f55c06bb32d013` (`upstream/main`).
Working branch: `client/dae-reproduction` in `Anamitra-Sarkar/DAE-method`.
Scope: Method=DAE, Dataset=BraTS2021 (`brats`), Input=128x128.

All paths below are relative to repo root. Verified by reading files inside the Codespace on 2026-09-08.

## 1. Entry points

- `reconstruction/train.py` — `get_method()` returns `DAEWorker` for `-m dae`; `main()` runs `Options(isTrain=True).parse() → save_options() → set_gpu_device → set_seed → set_network_loss → set_optimizer → set_dataloader → set_logging → run_train()`.
- `reconstruction/test.py` — `Options(isTrain=False).parse()` then `set_gpu_device → set_seed → set_network_loss → set_logging(test=True) → load_checkpoint → set_dataloader(test=True) → run_eval()`.
- `reconstruction/options.py` — argparse defaults: `-d rsna`, `-g 6`, `-m ae`, `--input-size 64`, `--base-width 16`, `--expansion 1`, `--hidden-num 1024`, `-ls 16`, `--en-depth 1`, `--de-depth 1`, `--train-batch-size 64`, `--train-lr 1e-3`, `--train-seed None` (random if unset). `epochs` map: `brats:250`. `in_c` map defaults to 1 except `c16:3`. `result_dir=~/Experiment/MedIAnomaly/<dataset>`, `save_dir=<result_dir>/dae/fold_<f>`, checkpoint at `<save_dir>/checkpoints/model.pt` (note: `save_checkpoint` writes `model.pt`, `load_checkpoint` reads `model.pt`; `options.py` default `test.model_path` string mentions `model.pth` but loader uses `model.pt` — actual file is `model.pt`).
- `reconstruction/train_eval.sh` — loops 7 datasets × methods on `gpu=7`, then DAE loop with `--input-size 128 -bs 16`. DAE commands reproduced (GPU id adapted):
  `python train.py -d brats -m dae -g <gpu> --input-size 128 -bs 16 -f <fold>`
  `python test.py -d brats -m dae -g <gpu> --input-size 128 -f <fold> -save`

## 2. Execution map (BraTS2021 + DAE)

```
BraTS2021 on disk (~/MedIAnomaly-Data/BraTS2021)
  train/ (normal only, filenames listed via os.listdir)
  test/normal/, test/tumor/, test/annotation/ (mask name = tumor name with flair->seg)
  ↓
reconstruction/dataloaders/dataload.py :: BraTSAD
  parallel_load (PIL, L mode since in_c=1, bilinear for images, nearest for masks)
  train returns {img,label=0,name}; test returns {img,label,name,mask}
  mask binarized at runtime: (np.array(mask)>0).astype(uint8)
  ↓
reconstruction/dataloaders/data_utils.py :: get_transform / get_data_path
  get_transform: ToTensor + Normalize((0.5,),(0.5,)) for in_c=1 → range [-1,1]
  get_data_path('brats') = ~/MedIAnomaly-Data/BraTS2021
  ↓
reconstruction/utils/dae_worker.py :: DAEWorker(AEWorker), noise_res=16, noise_std=0.2
  add_noise(x): N(0,0.2) at 16x16 → F.interpolate(bilinear, align_corners=True) to 128
    → torch.roll random (roll_x,roll_y) → if dataset in [brain,brats]: ns *= (x > x.min())
    → ns=(ns-0.5)*2 → noisy = x + ns
  train_epoch: noisy→net→criterion(clean, recon); Adam(lr=1e-3, wd=0)
  ↓
reconstruction/networks/unet.py :: UNet(in_channels=in_c, n_classes=in_c)
  depth=5, wf=6 (first width 2**6=64), ReflectionPad+WNConv2d+CustomSwish+GroupNorm blocks,
  avg_pool down, ConvTranspose2d up, skip concat, final 1x1 conv → {'x_hat': out}
  ↓
reconstruction/utils/losses.py :: AELoss
  train loss = mean((clean - x_hat)^2); anomaly_score=True returns mean over C (keepdim for maps)
  ↓
reconstruction/utils/ae_worker.py :: AEWorker.evaluate (inherited by DAE)
  test_loader bs=1; score_map = criterion(img, net_out, anomaly_score=True, keepdim=True)
  image score = mean over [C,H,W]; AUC/AP via sklearn; pixel path only if dataset==brats:
    PixAP/PixAUC over flattened maps vs masks; BestDice/BestThresh via compute_best_dice (200 thresholds, 8 procs)
  -save visualizes via visualize_2d: (img+1)/2, clamp, per-image min-max norm of map, overview = [img|x_hat|map|mask]
  ↓
~/Experiment/MedIAnomaly/brats/dae/fold_<f>/{train_options.txt, metrics.txt, checkpoints/model.pt, test_results/vis/...}
  wandb logging in set_logging (project MedIAnomaly) + thop FLOPs/params print
```

## 3. File-by-file notes

- `utils/base_worker.py::set_network_loss` — `dae` branch builds `UNet(in_c, in_c)` + `AELoss`, `.cuda()`. No SSIM/perceptual/L1 here.
- `utils/base_worker.py::set_dataloader` — `brats` uses `BraTSAD` both train and test; train bs=`opt.train[batch_size]`, test bs=1 shuffle=False.
- `utils/base_worker.py::set_seed` — uses `opt.train[seed]` else `randint(1,999999)`; sets random/numpy/torch/cuda seeds.
- `utils/base_worker.py::set_gpu_device` — `torch.cuda.set_device(opt.gpu)`; **must adapt/remove for CPU-only Codespace**.
- `utils/base_worker.py::set_logging` — `thop.profile` on CPU copy; `wandb.init`; prints config including num_params/FLOPs.
- `utils/ae_worker.py::pixel_metric` — True only for `brats`.
- `utils/util.py::compute_best_dice` — 200 thresholds from sorted unique scores, multiprocessing Pool(8).
- `networks/base_units/{ws_conv,swish}.py` — weight-standardized conv + custom Swish; not standard PyTorch U-Net.
- `dataloaders/dataload.py::BraTSAD` — single-channel (`L`), no modality stacking; resize in PIL before ToTensor.

## 4. Deviations / gotchas for reproduction

- Codespace has no GPU; upstream code calls `.cuda()` unconditionally. CPU run needs minimal device patch or GPU host (Kaggle P100).
- `train_eval.sh` hardcodes `gpu=7`; adapt to actual device (e.g. `-g 0` or CPU patch).
- Checkpoint filename is `model.pt` in code, not `model.pth` as hinted in `options.py` default string.
- `options.py` default `--input-size 64` but DAE loop uses `128`; reproduction uses `128`.
- `wandb` requires login; use `wandb offline` or disable for headless reproduction.
- `thop`, `medpy`, `SimpleITK`, `joblib` must be installed; `requirements.txt` from upstream to be pinned during env setup.
