#!/bin/bash
# DAE reproduction entrypoint — fails loudly. No `|| true` on critical steps.
# Usage: bash scripts/run_dae_reproduction.sh   (inside reconstruction/ parent)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== config =="
echo "method=dae dataset=brats input=128 batch=16 epochs=250 fold=0 seed=0"
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== dataset check =="
test -d ~/MedIAnomaly-Data/BraTS2021/train || { echo "missing ~/MedIAnomaly-Data/BraTS2021"; exit 1; }
echo "train: $(ls ~/MedIAnomaly-Data/BraTS2021/train | wc -l) (expect 4211)"

echo "== inspect =="
python3 scripts/inspect_brats.py | head -n 20

echo "== debug (noise/arch verification) =="
python3 scripts/debug_dae.py

echo "== train =="
export WANDB_MODE=offline
cd reconstruction
python3 train.py -d brats -m dae -g 0 --input-size 128 -bs 16 -f 0 --train-seed 0
echo "$PWD train.py -d brats -m dae -g 0 --input-size 128 -bs 16 -f 0 --train-seed 0" > ../artifacts/execution/final_command.txt

echo "== eval =="
python3 test.py -d brats -m dae -g 0 --input-size 128 -f 0 -save
cd ..

echo "== collect =="
EXP=~/Experiment/MedIAnomaly/brats/dae/fold_0
cp "$EXP/metrics.txt" artifacts/metrics/metrics.txt
cp "$EXP/checkpoints/model.pt" artifacts/checkpoints/model.pt
cp "$EXP/train_options.txt" artifacts/execution/train_options.txt
python3 scripts/verify_reproduction.py
echo "DONE"
