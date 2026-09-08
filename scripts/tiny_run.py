"""Tiny real-data validation run — PIPELINE VALIDATION RUN, NOT FINAL RESULT.
Uses real BraTS2021 + real DAE, 2 epochs, batch 8, seed 0.
"""
import sys
sys.path.insert(0, 'reconstruction')
sys.path.insert(0, 'scripts')
import cpu_shim  # noqa: patch .cuda on CPU
import os
os.environ.setdefault('WANDB_MODE', 'offline')
import torch
# remap cuda map_location to cpu for torch.load on CPU-only hosts
_orig_load = torch.load
def _cpu_load(*a, **k):
    m = k.get('map_location', None)
    if not torch.cuda.is_available() and isinstance(m, torch.device) and m.type == 'cuda':
        k['map_location'] = torch.device('cpu')
    return _orig_load(*a, **k)
torch.load = _cpu_load

from options import Options
from train import get_method

opt = Options(isTrain=True)
# emulate CLI: -d brats -m dae --input-size 128 -bs 8 -f valid_tiny --train-seed 0
sys.argv = ['train.py', '-d', 'brats', '-m', 'dae', '-g', '0',
            '--input-size', '128', '-bs', '8', '-f', 'valid_tiny', '--train-seed', '0']
opt.parse()
opt.train['epochs'] = 2
opt.train['eval_freq'] = 1
opt.save_options()
print(f"VALIDATION RUN fold={opt.fold} epochs={opt.train['epochs']} bs={opt.train['batch_size']} seed={opt.train['seed']}")
worker = get_method(opt)
worker.set_gpu_device()
worker.set_seed()
worker.set_network_loss()
worker.set_optimizer()
worker.set_dataloader()
worker.set_logging()
worker.run_train()
print('VALIDATION TRAIN DONE — running eval')
worker.set_dataloader(test=True)
worker.run_eval()
print('PIPELINE VALIDATION RUN COMPLETE — NOT FINAL RESULT')
