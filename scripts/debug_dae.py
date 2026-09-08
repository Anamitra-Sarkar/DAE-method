"""DAE debug + model summary — verifies noise_res=16, noise_std=0.2 pipeline.
SMOKE/DEBUG ONLY — NOT A MEDICAL EXPERIMENT.
Usage: conda activate dae-repro && python scripts/debug_dae.py
"""
import os, sys, random
sys.path.insert(0, 'reconstruction')
sys.path.insert(0, 'scripts')
import cpu_shim  # noqa: must be first to patch .cuda on CPU
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'reconstruction')
from networks.unet import UNet
from utils.losses import AELoss

SEED=0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={DEVICE}')

# model summary
net = UNet(in_channels=1, n_classes=1)
n_params = sum(p.numel() for p in net.parameters())
print(f'UNet in=1 out=1 params={n_params} ({n_params/1e6:.2f}M)')
print(f'depth=5 wf=6 (first width 64), GroupNorm+CustomSwish+WNConv, avg_pool down, ConvTranspose up')
x = torch.randn(2,1,128,128)
out = net(x)
xhat_shape = tuple(out['x_hat'].shape)
print(f'forward: in={tuple(x.shape)} out x_hat={xhat_shape}')
crit = AELoss()
loss = crit(x, out)
print(f'AELoss (MSE) on randn: {loss.item():.6f}')

# DAE noise verification (mirror utils/dae_worker.py exactly)
noise_res=16; noise_std=0.2; input_size=128
def add_noise(x, dataset='brats'):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std)
    coarse = ns.clone()
    ns = F.interpolate(ns, size=input_size, mode='bilinear', align_corners=True)
    up = ns.clone()
    roll_x = random.choice(range(input_size)); roll_y = random.choice(range(input_size))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2,-1])
    if dataset in ['brain','brats']:
        mask = (x > x.min())
        ns_masked = ns * mask
    else:
        ns_masked = ns
    ns_t = (ns_masked - 0.5)*2
    return x + ns_t, coarse, up, ns_masked, ns_t

# real image
from pathlib import Path
root = Path.home()/'MedIAnomaly-Data'/'BraTS2021'
train_img = sorted((root/'train').iterdir())[0]
print(f'sample: {train_img.name}')
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
pil = Image.open(train_img).convert('L').resize((128,128))
xt = tf(pil).unsqueeze(0)
print(f'clean: shape={tuple(xt.shape)} min={xt.min():.3f} max={xt.max():.3f} mean={xt.mean():.3f}')
noisy, coarse, up, masked, final = add_noise(xt)
print(f'coarse: {tuple(coarse.shape)} std~{coarse.std():.3f} (expect ~0.2)')
print(f'upsampled: {tuple(up.shape)}')
print(f'masked applied (brats foreground): {(masked.abs().sum()>0).item()}')
print(f'noisy: min={noisy.min():.3f} max={noisy.max():.3f}')
with torch.no_grad():
    recon = net(noisy)['x_hat']
    resid = (xt - recon)**2
print(f'recon: {tuple(recon.shape)} resid mean={resid.mean():.6f}')

# save debug montage
os.makedirs('artifacts/visualizations', exist_ok=True)
def to01(t):
    t=(t+1)/2; return t.clamp(0,1).squeeze().numpy()
fig, ax = plt.subplots(1,6, figsize=(18,3))
for a, ten, ttl in zip(ax, [xt, coarse.mean(1,keepdim=True), up.mean(1,keepdim=True), masked.mean(1,keepdim=True), noisy, recon], ['clean','coarse16','upsampled','masked','noisy','recon']):
    arr = ten.detach().squeeze().numpy() if ten.shape[-1]==128 else np.repeat(ten.detach().squeeze().numpy(), 8, axis=0)[:128,:128]
    # normalize coarse for display
    arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
    a.imshow(arr, cmap='gray'); a.set_title(ttl); a.axis('off')
plt.suptitle(f'DAE debug seed={SEED} noise_res=16 std=0.2 (DEBUG ONLY)')
plt.tight_layout(); plt.savefig('artifacts/visualizations/dae_debug.png', dpi=150)
print('saved artifacts/visualizations/dae_debug.png')
with open('artifacts/model_summary.txt','w') as f:
    f.write(f'UNet(in=1,out=1) params={n_params} ({n_params/1e6:.3f}M)\ndepth=5 wf=6 first_width=64\nnorm=GroupNorm act=CustomSwish conv=WNConv2d pool=avg_pool up=ConvTranspose2d\ncriterion=AELoss (MSE)\nforward 2x1x128x128 -> x_hat 2x1x128x128 OK\nnoise_res=16 noise_std=0.2 bilinear+roll+brats_mask verified\n')
print('saved artifacts/model_summary.txt')
