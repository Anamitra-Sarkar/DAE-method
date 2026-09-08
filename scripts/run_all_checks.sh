#!/bin/bash
# All checks: env + dataset + imports + model + smoke + outputs + results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[1] env"; python3 --version; python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo "[2] dataset"; python3 scripts/inspect_brats.py | head -n 12
echo "[3] imports"; python3 -c "import sys; sys.path.insert(0,'reconstruction'); from networks.unet import UNet; from utils.losses import AELoss; print('imports ok')"
echo "[4] model+noise debug"; python3 scripts/debug_dae.py | tail -n 8
echo "[5] outputs"; ls -la artifacts/dataset_manifest.json artifacts/model_summary.txt
echo "[6] results"; ls -la artifacts/results/ 2>/dev/null || echo "results pending (P100 run in progress)"
echo "CHECKS DONE"
