# Experiment Scope — DAE Reproduction for Client Delivery

## Decision

- **Method:** DAE only (`-m dae`). No AE/VAE/MemAE/CeAE/GANomaly/AE-U/grad variants.
- **Dataset:** BraTS2021 only (`-d brats`). No RSNA/VinDr/BrainTumor/LAG/ISIC/Camelyon16/OCT/Colon.
- **Input size:** `128×128` (`--input-size 128`), per `train_eval.sh` DAE loop.
- **Batch size:** `16` (`-bs 16`), per DAE loop.
- **Epochs:** `250` (repo default for `brats`), subject to feasibility check on CPU-only Codespace (2 cores, 8GB RAM, no GPU) / Kaggle P100 fallback.
- **Tasks:** anomaly classification (image AUC/AP) + anomaly segmentation (pixel AP/AUC/Dice) — both supported by `AEWorker.evaluate` when `dataset==brats`.
- **Fold:** `0` (`-f 0`) for single reproduction; five-fold only if time permits.

## Justification

1. Client request is DAE-specific; generic `train_eval.sh` covers 7 datasets × 12+ methods and is explicitly out of scope.
2. Supplied brief/slide identifies **BraTS2021 as brain MRI, normal-train / tumor-abnormal, with annotation masks for classification + segmentation** — the only dataset in scope with both heads.
3. Code has explicit BraTS handling: `BraTSAD` loader, `pixel_metric=True` only for `brats`, DAE foreground mask for `['brain','brats']`. This confirms BraTS2021 as the intended DAE target.
4. `train_eval.sh` gives DAE a dedicated loop with `128/-bs 16`, distinct from the 64px default for other methods — adopted verbatim (device id adapted).

## Non-goals

- Full 7-dataset benchmark.
- Architecture/hyperparameter tuning to improve scores.
- Accuracy as sole metric.
- Fake-data results reported as medical results (`make_fake_data.py` / smoke only for pipeline validation).

## Commands (device adapted; repo uses `gpu=7` which does not exist here)

```bash
cd reconstruction
python train.py -d brats -m dae -g <GPU> --input-size 128 -bs 16 -f 0
python test.py  -d brats -m dae -g <GPU> --input-size 128 -f 0 -save
```

On CPU-only Codespace the `.cuda()/set_device` calls require a minimal device patch or a GPU host (Kaggle P100 via `~/.kaggle/kaggle.json`). Any patch is documented in `docs/reproduction_report.md` § Deviations and does not change noise/architecture/loss.
