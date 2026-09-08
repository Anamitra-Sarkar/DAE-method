# Checkpoint — DAE BraTS2021 (P100, seed 0, 250 epochs)

The trained checkpoint is 119MB, over GitHub's 100MB single-file limit, so it is
published as a release asset rather than committed:

- Release: https://github.com/Anamitra-Sarkar/DAE-method/releases/tag/v0.1-dae-brats-p100
- File: `model.pt` (124,194,868 bytes)
- sha256: `d862a22413b75fe0e987dc4d0972d0aa9415135333ad968299d42144fa05505e`

Download:
```bash
gh release download v0.1-dae-brats-p100 --repo Anamitra-Sarkar/DAE-method \
  -p model.pt -O artifacts/checkpoints/model.pt
sha256sum artifacts/checkpoints/model.pt  # must match above
```

Training: UNet(1->1) on BraTS2021 (4211 normal, 128px, bs16, Adam lr 1e-3),
Tesla P100, torch 2.4.1+cu118, 12091.8s. Final metrics: AUC 0.8570, AP 0.9329,
PixAUC 0.9675, PixAP 0.7495, Dice 0.7058.
