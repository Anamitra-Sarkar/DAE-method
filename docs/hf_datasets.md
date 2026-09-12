# Hugging Face mirrors (private)

Account: `bhumika-tewari-282006`. All uploads ran on the Codespace from the
md5-verified Zenodo `MedIAnomaly-Data` tarballs (record 12677223); nothing
transited the local PC.

## Dataset repos (all private)

| Dataset | HF repo | Files | Zenodo md5 |
|---|---|---|---|
| BraTS2021 | `bhumika-tewari-282006/brats2021-medianomaly-dae` | 8938 | `237f2aee77e099f7483808ddec49a57c` |
| BrainTumor | `bhumika-tewari-282006/braintumor-medianomaly` | 2203 | `ab6482f646262fc4669b0135d80a0482` |
| LAG | `bhumika-tewari-282006/lag-medianomaly` | 3125 | `0b4034ae8d5275f7723a512adb218f86` |
| RSNA | `bhumika-tewari-282006/rsna-medianomaly` | 5855 | `5d6b85ce4b2daad8f1760eb23c050a7d` |
| VinCXR | `bhumika-tewari-282006/vincxr-medianomaly` | 6004 | `45cf848a4eba925062531c5f2d344ddc` |
| Camelyon16 | `bhumika-tewari-282006/camelyon16-medianomaly` | 7323 | `c26c026d1c315deb52720d4c195e874f` |

Each repo carries a dataset card (`README.md`) with license (CC-BY-4.0, as the
source record), structure, loader mapping, and citation.

## Model repo (private)

| Model | HF repo | Files |
|---|---|---|
| DAE BraTS2021 (seed 0, 250ep) | `bhumika-tewari-282006/dae-brats2021-medianomaly` | 4 (`pytorch_model.bin`, `config.json`, `README.md`, `.gitattributes`) |

## Not mirrored

ISIC2018 is not part of the Zenodo record — it requires a separate download
from the ISIC challenge archive (own account/terms), so it was left out.
