"""BraTS2021 dataset inspection — real data only.
Reports lengths, shapes, dtype, min/max, mean/std, channels, mask stats.
Usage: conda activate dae-repro && python scripts/inspect_brats.py [--save-dir artifacts/...]
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image

def load_png(path, mode='L'):
    return np.array(Image.open(path).convert(mode))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.expanduser('~/MedIAnomaly-Data/BraTS2021'))
    ap.add_argument('--save-dir', default='artifacts')
    ap.add_argument('--num-samples', type=int, default=8)
    args = ap.parse_args()
    root = Path(args.data_root)
    assert root.exists(), f'missing {root}'
    train_dir = root/'train'; tn = root/'test'/'normal'; tt = root/'test'/'tumor'; ta = root/'test'/'annotation'
    train = sorted(train_dir.iterdir()); normal = sorted(tn.iterdir()); tumor = sorted(tt.iterdir()); ann = sorted(ta.iterdir())
    print(f'train: {len(train)}')
    print(f'test normal: {len(normal)}')
    print(f'test tumor: {len(tumor)}')
    print(f'annotation: {len(ann)}')
    # sample stats
    def stats(paths, n=8, mask=False):
        arrs=[]
        for p in paths[:n]:
            a = load_png(str(p), 'L')
            arrs.append(a)
            print(f'  {p.name}: shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()} mean={a.mean():.2f} std={a.std():.2f} uniq={np.unique(a)[:8]}')
        return arrs
    print('--- train samples ---'); stats(train)
    print('--- test normal ---'); stats(normal)
    print('--- test tumor ---'); stats(tumor)
    print('--- annotation ---')
    for p in ann[:8]:
        a = load_png(str(p), 'L')
        print(f'  {p.name}: shape={a.shape} uniq={np.unique(a)} frac>0={(a>0).mean():.4f}')
    # alignment check
    tumor_names = {p.name for p in tumor}
    expected_masks = {n.replace('flair','seg') for n in tumor_names}
    actual_masks = {p.name for p in ann}
    print(f'mask alignment: expected={len(expected_masks)} actual={len(actual_masks)} missing={len(expected_masks-actual_masks)} extra={len(actual_masks-expected_masks)}')
    # transformed input check (ToTensor+Normalize(0.5,0.5))
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    im = Image.open(str(train[0])).convert('L').resize((128,128))
    t = tf(im)
    print(f'transformed: shape={tuple(t.shape)} dtype={t.dtype} min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f} std={t.std():.3f} channels={t.shape[0]}')
    os.makedirs(args.save_dir, exist_ok=True)
    manifest = {'train': len(train), 'test_normal': len(normal), 'test_tumor': len(tumor), 'annotation': len(ann)}
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
