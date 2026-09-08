"""Build client montage from test -save visualizations.
Expects <exp>/test_results/vis/overall/*.png named <label>_<name>.png (img|x_hat|map|mask).
Picks 2 healthy + 2 tumor samples, saves artifacts/visualizations/montage.png
plus per-sample panels. Usage: python scripts/make_montage.py --exp ~/Experiment/.../fold_0
"""
import argparse
from pathlib import Path
import random
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--outdir", default="artifacts/visualizations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    overall = Path(args.exp) / "test_results" / "vis" / "overall"
    files = sorted(overall.glob("*.png"))
    assert files, f"no viz in {overall}"
    healthy = [f for f in files if f.name.startswith("0_")]
    tumor = [f for f in files if f.name.startswith("1_")]
    pick = healthy[:2] + tumor[:2]
    assert len(pick) == 4, f"need 2+2 samples, got {len(healthy)} healthy {len(tumor)} tumor"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for f in pick:
        im = Image.open(f).convert("RGB")
        thumbs.append(im)
        im.save(out / f"sample_{f.stem}.png")
    W, H = thumbs[0].size
    canvas = Image.new("RGB", (W * 2, H * 2), "white")
    for i, t in enumerate(thumbs):
        canvas.paste(t, ((i % 2) * W, (i // 2) * H))
    canvas.save(out / "montage.png")
    print(f"saved {out}/montage.png from {[p.name for p in pick]}")

if __name__ == "__main__":
    main()
