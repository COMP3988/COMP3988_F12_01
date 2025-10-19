import argparse, os, time, numpy as np, torch
from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ResizeWithPadOrCropd, EnsureTyped
from monai.inferers import SlidingWindowInferer
from skimage.metrics import structural_similarity as ssim

from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler
from network.Diffusion_model_transformer import SwinVITModel

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, required=True)
p.add_argument("--ckpt", type=str, required=True)
p.add_argument("--eval_steps", type=int, default=10)
p.add_argument("--sw_batch_size", type=int, default=32)
p.add_argument("--overlap", type=float, default=0.0)
p.add_argument("--max_cases", type=int, default=0)
args = p.parse_args()

# ---------------- config (match training) ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size   = (256,256,128)
patch_size = (64,64,2)
patch_num  = 1

# diffusion (match training)
diffusion_steps=1000
learn_sigma=True
sigma_small=False
noise_schedule='linear'
use_kl=False
predict_xstart=True
rescale_timesteps=True
rescale_learned_sigmas=True

def make_eval_diffusion(num_steps):
    return create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=[num_steps],
    )

# model (match training exactly)
num_channels=64
attention_resolutions="32,16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4,2],[4,4,2],[4,4,2],[4,4,2]]
num_res_blocks = [1,1,1,1]
sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),  # keep trailing comma to match training
attention_ds = [int(x) for x in attention_resolutions.split(",")]

model = SwinVITModel(
    image_size=patch_size,
    in_channels=2,
    model_channels=num_channels,
    out_channels=2,
    dims=3,
    sample_kernel=sample_kernel,
    num_res_blocks=num_res_blocks,
    attention_resolutions=tuple(attention_ds),
    dropout=0,
    channel_mult=channel_mult,
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=num_heads,
    window_size=window_size,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=False,
).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
model.eval()

# ---------------- dataset ----------------
class NpzSet(Dataset):
    def __init__(self, root):
        self.files = natsorted([str(p) for p in Path(root).glob("*.npz")])
        if not self.files: raise FileNotFoundError(f"No .npz in {root}")
        self.tx = Compose([
            EnsureChannelFirstd(keys=["image","label"], channel_dim="no_channel"),
            ResizeWithPadOrCropd(keys=["image","label"], spatial_size=img_size, constant_values=-1),
            EnsureTyped(keys=["image","label"]),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        with np.load(f) as d:
            mri = d["image"].astype(np.float32)
            ct  = d["label"].astype(np.float32)
        out = self.tx({"image": mri, "label": ct})
        return out["image"].to(torch.float32), out["label"].to(torch.float32), f

test_ds = NpzSet(args.data_dir)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- inference helpers ----------------
eval_diff = make_eval_diffusion(args.eval_steps)
inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=args.sw_batch_size,
                               overlap=args.overlap, mode="constant")

@torch.no_grad()
def diffusion_sampling_with(diffusion_obj, condition, model):
    return diffusion_obj.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,
        clip_denoised=True,
    )

def vol_ssim(pred, gt):
    # pred, gt: numpy volumes in [-1,1], shape [D,H,W]
    # try full 3D SSIM; fall back to slice-wise if needed
    try:
        return ssim(gt, pred, data_range=2.0, win_size=7)
    except Exception:
        ss = []
        for z in range(pred.shape[-1]):
            ss.append(ssim(gt[...,z], pred[...,z], data_range=2.0, win_size=7))
        return float(np.mean(ss))

# ---------------- run ----------------
ssims = []
t0 = time.time()
for idx, (img, lbl, path) in enumerate(test_loader):
    if args.max_cases and idx >= args.max_cases: break
    img = img.to(device)   # [B=1, C=1, D,H,W]
    lbl = lbl.to(device)

    # make condition+model 2-channel like training used (in_channels=2)
    condition = img  # MRI as condition
    # Sliding window + diffusion
    sampled = inferer(
        condition,
        lambda c, m: diffusion_sampling_with(eval_diff, c, m),
        model,
    )

    # SSIM expects single channel arrays
    pred_np = sampled[0,0].detach().cpu().numpy()
    gt_np   = lbl[0,0].detach().cpu().numpy()
    s = vol_ssim(pred_np, gt_np)
    ssims.append(s)
    print(f"{Path(path[0]).name}: SSIM={s:.6f}")

mean_ssim = float(np.mean(ssims)) if ssims else float("nan")
print(f"Cases={len(ssims)}  Mean SSIM={mean_ssim:.6f}  Time={time.time()-t0:.2f}s")