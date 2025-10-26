import os
from pathlib import Path
import argparse
import numpy as np
import torch
import SimpleITK as sitk

# reuse your UNet
from model import UNet
# reuse helpers from predict.py
from predict import _array2pil_uint8, predict_img

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

@torch.inference_mode()
def predict_volume_with_predict_py(net, mr_path, device, scale):
    """Use predict.py helpers to synthesize CT volume from mr.mha."""
    mr_itk = sitk.ReadImage(str(mr_path))
    vol = sitk.GetArrayFromImage(mr_itk).astype(np.float32)  # (D,H,W) or (H,W)
    if vol.ndim == 2:
        vol = vol[None, ...]
    preds = []
    for z in range(vol.shape[0]):
        pil_slice = _array2pil_uint8(vol[z])
        pred_slice = predict_img(net=net,
                                 full_img=pil_slice,
                                 device=device,
                                 scale_factor=scale,
                                 out_threshold=0.5)
        preds.append(pred_slice.astype(np.float32))
    pred = np.stack(preds, axis=0)  # (D,H,W), float
    out_itk = sitk.GetImageFromArray(pred)
    out_itk.CopyInformation(mr_itk)
    return out_itk

def take_last_reverse(dirpath: Path, k: int):
    names = sorted([p.name for p in dirpath.iterdir() if p.is_dir()], reverse=True)
    return names[:k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help=".../Task1 containing AB/HN/TH")
    ap.add_argument("--checkpoint", required=True, help="UNet .pth")
    ap.add_argument("--out-dir", default="./demo_results/images")
    ap.add_argument("--per-section", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    net = UNet(n_channels=1, n_classes=1, bilinear=True).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state, strict=True)

    regions = ["AB", "HN", "TH"]
    for region in regions:
        rdir = root / region
        if not rdir.is_dir():
            continue
        for patient in take_last_reverse(rdir, args.per_section):
            pdir = rdir / patient
            mr = pdir / "mr.mha"
            ct = pdir / "ct.mha"
            if not (mr.exists() and ct.exists()):
                continue

            # synthesize CT (this is the "fake")
            pred_img = predict_volume_with_predict_py(net, mr, device, args.scale)

            fake_name = f"{region}_{patient}_fake_B.mha"
            real_name = f"{region}_{patient}_real_A.mha"
            sitk.WriteImage(pred_img, str(out_dir / fake_name))
            sitk.WriteImage(sitk.ReadImage(str(ct)), str(out_dir / real_name))
            print(f"Wrote {fake_name} and {real_name}")

if __name__ == "__main__":
    main()
