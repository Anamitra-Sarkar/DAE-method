"""Panels + montage from the REAL P100 checkpoint on CPU (device-only shim).
Fixed samples: 2 healthy + 2 tumor. Saves clean | recon | residual | mask panels
and a montage. Usage: python scripts/make_panels.py --ckpt artifacts/checkpoints/model.pt
"""
import argparse, sys
sys.path.insert(0, "reconstruction")
sys.path.insert(0, "scripts")
import cpu_shim  # noqa: CPU patch first
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_img(p, size=128):
    im = Image.open(p).convert("L").resize((size, size))
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return tf(im)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", default="artifacts/visualizations")
    args = ap.parse_args()
    from networks.unet import UNet
    root = Path(args.data_root or (str(Path.home() / "MedIAnomaly-Data" / "BraTS2021")))
    normals = sorted((root / "test" / "normal").iterdir())
    tumors = sorted((root / "test" / "tumor").iterdir())
    picks = [("healthy", normals[0]), ("healthy", normals[1]),
             ("tumor", tumors[0]), ("tumor", tumors[1])]
    net = UNet(in_channels=1, n_classes=1)
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for r, (kind, p) in enumerate(picks):
        x = load_img(str(p)).unsqueeze(0)
        with torch.no_grad():
            xhat = net(x)["x_hat"]
        resid = (x - xhat) ** 2
        score = resid.mean().item()
        mask = None
        if kind == "tumor":
            mp = root / "test" / "annotation" / p.name.replace("flair", "seg")
            mask = np.array(Image.open(mp).convert("L").resize((128, 128))) > 0
        def to01(t):
            return ((t + 1) / 2).clamp(0, 1).squeeze().numpy()
        panels = [to01(x), to01(xhat), resid.squeeze().numpy(), mask]
        titles = [f"{kind} clean", "recon", f"resid mean={score:.4f}", "mask"]
        for c, (arr, t) in enumerate(zip(panels, titles)):
            ax = axes[r][c]
            if arr is None:
                ax.text(0.5, 0.5, "n/a (healthy)", ha="center"); ax.axis("off"); continue
            a = np.asarray(arr, dtype=float)
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            ax.imshow(a, cmap="gray"); ax.set_title(f"{p.name[:22]} {t}", fontsize=7); ax.axis("off")
        Image.fromarray((to01(xhat) * 255).astype(np.uint8)).save(out / f"panel_{p.stem}_rec.png")
    plt.suptitle("DAE P100 checkpoint panels (real data, CPU render)")
    plt.tight_layout()
    plt.savefig(out / "montage.png", dpi=150)
    print(f"saved {out}/montage.png + per-sample rec panels")

if __name__ == "__main__":
    main()
