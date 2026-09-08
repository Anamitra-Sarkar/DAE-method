# DAE-method — Denoising Autoencoder for Brain-MRI Anomaly Detection

Reproducible DAE anomaly-detection experiment on real BraTS2021 data:
healthy-only training, coarse-noise denoising, customized U-Net,
reconstruction-error scoring (image-level detection + pixel-level localization).

**Start here: [`CLIENT_README.md`](CLIENT_README.md)** — objective, install,
dataset, training, evaluation, results, file map.
**Setup guide: [`docs/SETUP.md`](docs/SETUP.md)** — every step exactly as built.

Final result (Tesla P100, seed 0, 250 epochs): AUC 0.8570, AP 0.9329,
PixAUC 0.9675, PixAP 0.7495, Dice 0.7058. See
[`artifacts/results/final_results.md`](artifacts/results/final_results.md).

License: MIT — see [LICENSE](LICENSE).
