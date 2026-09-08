"""Functional correctness tests with real data + real checkpoint (examples).
Run: conda activate dae-repro && python scripts/test_functional.py [--ckpt PATH]
Exit non-zero on any failure. No fake medical results; asserts only.
"""
import argparse, sys
sys.path.insert(0, "reconstruction")
sys.path.insert(0, "scripts")
import cpu_shim  # noqa: CPU patch first
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import random

PASS = []
def check(name, cond, detail=""):
    assert cond, f"FAIL {name}: {detail}"
    PASS.append(name)
    print(f"pass: {name} {detail}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-root", default=str(Path.home() / "MedIAnomaly-Data" / "BraTS2021"))
    args = ap.parse_args()
    root = Path(args.data_root)

    # 1. dataset example counts (observed ground truth)
    n_train = len(list((root / "train").iterdir()))
    n_norm = len(list((root / "test" / "normal").iterdir()))
    n_tum = len(list((root / "test" / "tumor").iterdir()))
    n_ann = len(list((root / "test" / "annotation").iterdir()))
    check("counts", (n_train, n_norm, n_tum, n_ann) == (4211, 828, 1948, 1948),
          f"{n_train}/{n_norm}/{n_tum}/{n_ann}")

    # 2. mask alignment example: first tumor file has matching seg mask
    t0 = sorted((root / "test" / "tumor").iterdir())[0]
    m0 = root / "test" / "annotation" / t0.name.replace("flair", "seg")
    check("mask_exists", m0.exists(), str(m0.name))
    mu = np.unique(np.array(Image.open(m0).convert("L")))
    check("mask_binary", set(mu.tolist()) <= {0, 255}, str(mu))

    # 3. transform example: ToTensor+Normalize gives 1x128x128 in [-1,1]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    x = tf(Image.open(root / "train" / sorted((root / "train").iterdir())[0].name).convert("L").resize((128, 128)))
    check("transform", tuple(x.shape) == (1, 128, 128) and float(x.min()) >= -1.0 and float(x.max()) <= 1.0,
          f"{tuple(x.shape)} [{float(x.min()):.2f},{float(x.max()):.2f}]")

    # 4. DAE noise example: coarse 16x16 std~0.2, upsampled 128, masked, shifted
    random.seed(0); torch.manual_seed(0)
    ns = torch.normal(mean=torch.zeros(1, 1, 16, 16), std=0.2)
    check("noise_std", abs(float(ns.std()) - 0.2) < 0.05, f"std={float(ns.std()):.3f}")
    up = F.interpolate(ns, size=128, mode="bilinear", align_corners=True)
    check("noise_upsample", tuple(up.shape) == (1, 1, 128, 128), str(tuple(up.shape)))

    # 5. UNet example: forward shape + param count
    from networks.unet import UNet
    net = UNet(in_channels=1, n_classes=1)
    n_params = sum(p.numel() for p in net.parameters())
    check("unet_params", n_params == 31042369, str(n_params))
    with torch.no_grad():
        out = net(torch.randn(1, 1, 128, 128))["x_hat"]
    check("unet_forward", tuple(out.shape) == (1, 1, 128, 128), str(tuple(out.shape)))

    # 6. checkpoint example (if provided): loads, tumor residual > healthy residual
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        def resid_mean(p):
            t = tf(Image.open(p).convert("L").resize((128, 128))).unsqueeze(0)
            with torch.no_grad():
                return float(((t - net(t)["x_hat"]) ** 2).mean())
        rh = resid_mean(sorted((root / "test" / "normal").iterdir())[3])
        rt = resid_mean(sorted((root / "test" / "tumor").iterdir())[10])
        check("ckpt_discriminates", rt > rh, f"tumor={rt:.5f} > healthy={rh:.5f}")
    else:
        print("skip: ckpt_discriminates (no --ckpt)")

    print(f"\nALL {len(PASS)} TESTS PASSED")

if __name__ == "__main__":
    main()
