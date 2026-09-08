# DAE BraTS2021 final results
Method=DAE Dataset=BraTS2021 Input=128 Seed=0 Fold=0
Reference: medianomaly.pdf Table 6 (p.11, image) + Table 7 (p.12, pixel); paper = mean+-std over 3 seeds, reproduction = seed 0.

| Metric | Paper term | Published | Reproduced | Diff | |Diff| | % diff | Source |
|---|---|---|---|---|---|---|---|
| AUC | AUC | 0.859+-0.010 | 0.85699 | -0.00201 | 0.00201 | 0.23% | Table 6 (image-level AnoCls) p.11 |
| AP | AP (image) | 0.934+-0.005 | 0.93286 | -0.00114 | 0.00114 | 0.12% | Table 6 (image-level AnoCls) p.11 |
| PixAUC | pixel AUC (not reported) | N/A (excluded by paper, p.9) | 0.96751 | N/A | N/A | N/A | medianomaly.pdf p.9 (pixel AUC not employed) |
| PixAP | APpix | 0.755+-0.007 | 0.74949 | -0.00551 | 0.00551 | 0.73% | Table 7 (pixel-level AnoSeg) p.12 |
| BestDice | ⌈Dice⌉ | 0.711+-0.006 | 0.70583 | -0.00517 | 0.00517 | 0.73% | Table 7 (pixel-level AnoSeg) p.12 |
| BestThresh | — | N/A (no published counterpart) | 0.06802 | N/A | N/A | N/A | — |
| normal_score | — | N/A (no published counterpart) | 0.00275 | N/A | N/A | N/A | — |
| abnormal_score | — | N/A (no published counterpart) | 0.01211 | N/A | N/A | N/A | — |
