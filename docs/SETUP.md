# SETUP — DAE BraTS2021 reproduction, exactly as built

Every command below was executed for real during this delivery (Codespace
`cuddly-enigma-jj9jvwxxqgpr35v4w` + Kaggle kernel `arkosarkarhehe/dae-brats-p100-train`).
No step is aspirational. Tested 2026-09-08.

## 0. Prerequisites

- GitHub account with access to `Anamitra-Sarkar/DAE-method`.
- GitHub CLI (`gh`) authenticated with `repo` + `codespace` scopes:
  `gh auth refresh -h github.com -s codespace` (a token lacking the
  `codespace` scope fails with `HTTP 403` on `gh codespace list`).
- Kaggle account + `~/.kaggle/kaggle.json` for the GPU run (P100).

## 1. Open the Codespace

```bash
gh codespace list   # expect cuddly-enigma-jj9jvwxxqgpr35v4w / Available
gh codespace ssh --codespace cuddly-enigma-jj9jvwxxqgpr35v4w -- "echo AWAKE"
```

Machine: 2 vCPU, 8GB RAM, ~19GB free on `/workspaces`, **CPU-only**
(`nvidia-smi` absent). Python 3.14 default is TOO NEW — use the conda env below.

Inside the Codespace the delivery repo lives at `/workspaces/DAE-method`:

```bash
cd /workspaces/DAE-method
git remote -v   # origin=Anamitra-Sarkar/DAE-method
git checkout main
git log -1 --oneline
```

## 2. Python environment (CPU Codespace)

```bash
conda create -y -n dae-repro python=3.10
source /opt/conda/etc/profile.d/conda.sh && conda activate dae-repro
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib wandb thop medpy scikit-image \
  SimpleITK joblib pandas scipy openpyxl
pip install 'numpy<2'   # REQUIRED: torch 2.1.2 is built against numpy 1.x;
                        # numpy 2.x crashes with "compiled using NumPy 1.x"
export WANDB_MODE=offline   # repo logs to wandb; offline keeps it headless
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.1.2+cpu False
```

Record (as this delivery does under `artifacts/execution/`):

```bash
python --version | tee artifacts/execution/torch_info.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" \
  | tee -a artifacts/execution/torch_info.txt
pip freeze | tee artifacts/execution/pip_freeze.txt
```

## 3. Dataset (BraTS2021 only — NOT all seven)

```bash
mkdir -p ~/MedIAnomaly-Data && cd ~/MedIAnomaly-Data
wget -q --show-progress -O BraTS2021.tar.gz \
  https://zenodo.org/api/records/12677223/files/BraTS2021.tar.gz/content
echo "237f2aee77e099f7483808ddec49a57c  BraTS2021.tar.gz" | md5sum -c -
tar -xzf BraTS2021.tar.gz
ls BraTS2021/train | wc -l            # 4211
ls BraTS2021/test/normal | wc -l      # 828
ls BraTS2021/test/tumor | wc -l       # 1948
ls BraTS2021/test/annotation | wc -l  # 1948
```

Audit the loader against the real files:

```bash
cd /workspaces/DAE-method
python scripts/inspect_brats.py
# single-channel 208px PNGs, masks 0/255, flair->seg alignment 100%,
# transformed model input 1x128x128 in [-1,1]
```

## 4. Verify DAE mechanics before training (CPU, ~1 min)

```bash
python scripts/debug_dae.py
# UNet in=1 out=1 params=31042369 (31.04M), forward 2x1x128x128 OK,
# coarse noise 16x16 std~0.2, bilinear+roll+brats foreground mask verified.
# Writes artifacts/model_summary.txt + artifacts/visualizations/dae_debug.png
# (DEBUG ONLY — never a medical result).
```

Note: `set_logging` prints `num_params: 2.7863M` — a **thop undercount**
(thop registers custom `WNConv2d` as `zero_ops`). `sum(p.numel())` = 31.04M
is the true count.

## 5. CPU device shim (Codespace only)

The reconstruction code calls `.cuda()` unconditionally. On CPU-only hosts the repo ships a
**device-only** shim that changes no math (`scripts/cpu_shim.py`: `.cuda()` and
`set_device/seed` become no-ops, `torch.load` cuda→cpu remap). GPU runs import
nothing and execute unmodified code. Do NOT "fix" noise/architecture/loss.

## 6. Full training (GPU — Kaggle P100)

The CPU Codespace provably cannot train this (tiny 2-epoch validation loaded all
4211 images fine but could not finish one epoch in 25 min). Full run used a
Kaggle script kernel (P100 is Pascal sm_60 → torch **cu118**, not the default
CUDA build, which lacks sm_60 kernels):

```bash
# local machine with kaggle.json:
kaggle kernels push -p kaggle-dae   # kernel-metadata.json: enable_gpu + internet
kaggle kernels status arkosarkarhehe/dae-brats-p100-train
```

Kernel steps (`kaggle-dae/dae-p100-train.py` pattern — kept out of git because it
is infra glue, logic mirrored here): install `torch==2.4.1+cu118`, clone the repo (public, no token), download+md5 BraTS2021, then:

```bash
cd repro/reconstruction
python train.py -d brats -m dae -g 0 --input-size 128 -bs 16 -f 0 --train-seed 0
python test.py  -d brats -m dae -g 0 --input-size 128 -f 0 -save
```

Never use the repo default `-g 7` (hardware that does not exist here) and never
run `./train_eval.sh` (full 7-dataset benchmark — out of scope).

Result of record: train 12091.8s, eval 153.2s, seed 0 → AUC 0.8570, AP 0.9329,
PixAUC 0.9675, PixAP 0.7495, Dice 0.7058. Raw: `artifacts/metrics/metrics.txt`.

## 7. Checkpoint (119MB — release asset, NOT git)

```bash
gh release download v0.1-dae-brats-p100 --repo Anamitra-Sarkar/DAE-method \
  -p model.pt -O artifacts/checkpoints/model.pt
sha256sum artifacts/checkpoints/model.pt
# d862a22413b75fe0e987dc4d0972d0aa9415135333ad968299d42144fa05505e
```

## 8. Results, panels, verification

```bash
python scripts/make_results.py --metrics artifacts/metrics/metrics.txt \
  --outdir artifacts/results          # csv + md + xlsx, real values only
python scripts/make_panels.py --ckpt artifacts/checkpoints/model.pt
# artifacts/visualizations/montage.png (2 healthy + 2 tumor, clean|recon|resid|mask)
python scripts/verify_reproduction.py # exit 0 = complete
bash scripts/run_all_checks.sh        # env+data+imports+model+outputs+results
bash scripts/run_dae_reproduction.sh  # full rerun entrypoint (needs GPU + data)
```

## 9. Troubleshooting (all observed first-hand)

| Symptom | Cause | Fix |
|---|---|---|
| `gh codespace list` → HTTP 403 | token lacks `codespace` scope | `gh auth refresh -s codespace` or classic token with it |
| `gpg: signing failed: No secret key` on commit | Codespace gitconfig enables signing | `git config commit.gpgsign false` (+ `--global`) |
| `git push` → `could not read Username` | git credential helper not linked to `gh` auth | `gh auth login --with-token` then `gh auth setup-git` |
| `ModuleNotFoundError: torch` | system python 3.14 has no torch | `conda activate dae-repro` first |
| `compiled using NumPy 1.x` crash | numpy 2.x with torch 2.1.2 | `pip install 'numpy<2'` |
| P100 `no kernel image available` / CUDA errors | default torch lacks sm_60 | torch cu118 build (`2.4.1+cu118` proven) |
| `kaggle kernels output` stops at 353 files | output listing pagination | viz regenerated via `make_panels.py` from release checkpoint |
| No metric lines for ~20 min mid-training | eval only at ep1 + every 25 | expected; do not kill the run |
