# DAE-method — Denoising Autoencoder for Brain-MRI Anomaly Detection

Reproducible DAE anomaly-detection experiment on real BraTS2021 data:
healthy-only training, coarse-noise denoising, customized U-Net,
reconstruction-error scoring (image-level detection + pixel-level localization).

**Full guide: [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md)** — objective,
installation, dataset, training, evaluation, results, file map.
**Setup, exactly as built: [`docs/SETUP.md`](docs/SETUP.md).**

Final result (Tesla P100, seed 0, 250 epochs): AUC 0.8570 vs published
0.859±0.010; Dice 0.7058 vs 0.711±0.006 — near-exact reproduction.
Details: [`artifacts/results/final_results.md`](artifacts/results/final_results.md).

License: MIT — see [LICENSE](LICENSE).
