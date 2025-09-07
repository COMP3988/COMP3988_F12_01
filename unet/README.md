Adapted from https://github.com/milesial/Pytorch-UNet/

# MRI → CT Synthesis (U-Net)

## Overview
Supervised MRI→CT synthesis using a U-Net adapted for SynthRAD2025.

- Train on `.mha` MR/CT pairs
- Predict single volumes
- Batch synthesize CTs and pair with ground truth
- Evaluate with MAE/SSIM/PSNR

## Requirements
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ensure these are present if not in requirements:
# pip install torch SimpleITK numpy scikit-image pillow matplotlib tqdm
```

## Data Layout

```
synthRAD2025_Task1_Train/Task1/
  AB/<PATIENT>/{mr.mha, ct.mha}
  HN/<PATIENT>/{mr.mha, ct.mha}
  TH/<PATIENT>/{mr.mha, ct.mha}
```

## Training

```bash
python3 unet/train.py \
  --data-root ./synthRAD2025_Task1_Train/Task1 \
  --target ct \
  --epochs 50 \
  --batch-size 1 \
  --scale 1.0 \
  --samples-per-section 0
# checkpoints saved to ./checkpoints/
```

## Predict a Single Volume

```bash
python3 unet/predict.py \
  --model checkpoints/checkpoint_epoch20.pth \
  --input synthRAD2025_Task1_Train/Task1/AB/1ABA005/mr.mha \
  --output out.mha \
  --scale 1.0
```

Outputs a predicted CT volume with original geometry prserved.

## Batch Synthesize + Pair (Fake vs Real)

Generates predicted CTs (fake) from mr.mha and copies real CTs (real) into one folder. Takes the last X patients per region in reverse alphabetical order.

```bash
python3 unet/predict_batch_set.py \
  --data-root ./synthRAD2025_Task1_Train/Task1 \
  --checkpoint ./checkpoints/checkpoint_epoch20.pth \
  --per-section 20 \
  --out-dir ./demo_results/unet_images \
  --scale 1.0
```

Output naming in ./demo_results/unet_images/:

```
AB_<PATIENT>_fake_B.mha   # predicted CT (from MR)
AB_<PATIENT>_real_A.mha   # ground-truth CT
(HN/TH similarly)
```

## Evaluate (MAE/SSIM/PSNR)

Run evaluation on the paired .mha outputs produced above.

```bash
python3 unet/evaluate.py \
  --results-dir ./demo_results/unet_images
```


- Loads *_fake_B.mha and *_real_A.mha
- Computes per-slice MAE, SSIM, PSNR with per-slice 0-1 normalization
- Prints per-patient, per-region, and overall summaries
- Writes summary.json (and plots if enabled)

## Devices

Automatic: CUDA → MPS (Apple Silicon) → CPU. Use --amp for autocast inference.

## Troubleshooting
- Non-writable NumPy warning: harmless; tensors are copied as needed.
- Shape mismatch on load: checkpoint must match the U-Net config used for training.
- No outputs: check data layout and existence of both mr.mha and ct.mha per patient.

## Repo Structure (relevant)

```
unet/
  model/
    unet_model.py
    unet_parts.py
  train.py
  predict.py               # single-volume MR→CT
  predict_batch_set.py     # batch MR→CT → *_fake_B.mha / *_real_A.mha
  evaluate.py              # metrics over paired .mha volumes
requirements.txt
```

## Notes

Metrics reported: **MAE** (pixel), **SSIM** (structure), **PSNR** (quality)

