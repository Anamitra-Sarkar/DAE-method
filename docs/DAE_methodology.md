# DAE Methodology (for Bhumika)

## 1. Problem
Unsupervised medical anomaly detection: given only healthy images at training time,
flag abnormal images at test time (image-level classification) and localize lesions
(pixel-level segmentation). BraTS2021: train = normal FLAIR slices; test = normal +
tumor slices with binary masks.

## 2. Normal-only training
The model never sees tumors in training, so it learns the manifold of healthy anatomy.
At test time, tumor regions fall off-manifold and reconstruct poorly.

## 3. Coarse noise injection (the DAE trick)
For each clean image `x` (normalized to [-1,1]):

```
ns0 ~ N(0, 0.2^2)                      # 16x16, per-sample per-channel
ns1 = BilinearUpsample(ns0 -> 128x128) # coarse -> smooth blotches
ns2 = Roll(ns1, random dx, dy)         # random translation
ns3 = ns2 * (x > min(x))               # BraTS/Brain foreground mask only
ns  = (ns3 - 0.5) * 2
x_noisy = x + ns
```

Why coarse? Full-resolution white noise is trivially removable by blur; coarse
blotches mimic lesion-scale corruption and force semantic inpainting.
Why the mask? MRI backgrounds are uniformly black; noising them wastes capacity
and leaks a background-reconstruction shortcut. Masking restricts corruption (and
hence learning) to tissue.

## 4. Denoising objective
Customized U-Net `f` reconstructs the CLEAN image from the noisy one:

```
L = mean((f(x_noisy) - x)^2)    # AELoss, plain MSE
```

No SSIM/perceptual/KL terms for DAE. Target is always the clean original.

## 5. U-Net (MedIAnomaly variant)
`UNet(in=1, out=1, depth=5, wf=6)`: first width 64, doubling per level;
ReflectionPad + weight-standardized conv + CustomSwish + GroupNorm blocks;
average-pool down; transposed-conv up with skip concatenation; final 1x1 conv.
~31.04M params. Skip connections preserve healthy fine detail so residuals
concentrate on true anomalies rather than blur.

## 6. Scoring
Pixel map: `s = mean_c((x - x_hat)^2)` (Nx1xHxW). Image score: spatial mean of `s`.
Test-time input is CLEAN (no added noise); tumors reconstruct badly -> high `s`.

## 7. Evaluation (BraTS only has pixel heads)
- Image: AUC, AP over image scores vs labels.
- Pixel: PixAUC/PixAP over flattened maps vs masks; BestDice over 200 thresholds.
Accuracy is reported only supplementally, never as the primary metric.

## 8. Implementation <-> paper
Repo pins `noise_res=16, noise_std=0.2`, bilinear+roll+foreground mask, 128px input,
MSE loss, single-channel FLAIR. Any textbook DAE default (other noise scale,
BatchNorm/ReLU U-Net, L1/SSIM loss) must NOT be silently substituted.
See `docs/paper_vs_repository.md` for the source table.
