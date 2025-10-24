# Transformer-Diffusion Model for MRI→CT Synthesis

Originally from [this repo](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/blob/main/README.md?plain=1)

A 3D transformer-based denoising diffusion model for synthetic CT generation from MRI scans.

## Environment Setup

Target environment: **Python 3.10** with **CUDA 12.1** (tested on NVIDIA L4 GPU).
We recommend using **mamba** or **micromamba** for dependency resolution.

### Install micromamba (optional)

```bash
curl micro.mamba.pm/install.sh | bash
```

Restart your shell or source your `~/.bashrc` / `~/.zshrc`.

## Create environment

```bash
mamba create -n synthrad-l4 python=3.10
mamba activate synthrad-l4
```

Install PyTorch with CUDA

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install supporting packages

```bash
mamba install -c conda-forge monai einops timm natsort scikit-image scipy tqdm matplotlib
```

If you hit issues with `pkg_resources` deprecation

```bash
pip install "setuptools<81"
```

Verify gpu

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Preprocessing

Preprocess SynthRAD dataset into `.npz` files:

```bash
python preprocess_synthrad.py --synthrad_location <path-to-synthrad>
```

This generates normalized MRI/CT `.npz` files under:
- `SynthRAD/imagesTr` (training)
- `SynthRAD/imagesVal` (validation)
- `SynthRAD/imagesTs` (testing)

Each `.npz` file contains:
- `image`: MRI volume (normalized to [-1,1])
- `label`: CT volume (normalized to [-1,1])

## Training

Run the main training script:

```bash
python main.py --epochs 50
```

### Training Features

- Configurable epochs via `--epochs` argument
- tqdm progress bars for training and evaluation
- MONAI transforms with `channel_dim="no_channel"`
- Sliding-window inference for evaluation
- Automatic checkpoint saving

## Inference

### Single File Inference

Process a single MRI volume:

```bash
python infer_single.py \
  --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
  --input SynthRAD/imagesTs/patient001.npz \
  --output patient001_ct_pred.mha \
  --steps 20 --overlap 0.25 --sw-batch 16
```

### Batch Directory Inference

Process all files in a directory:

```bash
python infer_directory.py \
  --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
  --inputdir SynthRAD/imagesTs \
  --outputdir outputs_mha \
  --steps 20 --overlap 0.25 --sw-batch 16
```

### Inference Parameters

- `--steps`: Number of diffusion steps (default: 2)
- `--overlap`: Sliding-window overlap ratio [0,1) (default: 0.5)
- `--sw-batch`: Sliding-window batch size (default: 8)
- `--fp16`: Enable mixed precision inference
- `--device`: Device to use (default: cuda:0)

## File Structure

```
shaoyanpan-transformer-diffusion/
├── main.py                    # Training script
├── infer_single.py           # Single file inference
├── infer_directory.py        # Batch directory inference
├── preprocess_synthrad.py    # SynthRAD preprocessing
├── diffusion/                # Diffusion model components
├── network/                  # Transformer network architecture
├── SynthRAD/                 # Preprocessed data (created by preprocessing)
│   ├── imagesTr/            # Training .npz files
│   ├── imagesVal/           # Validation .npz files
│   └── imagesTs/            # Testing .npz files
└── synthRAD_checkpoints/    # Model checkpoints
    └── Synth_A_to_B.pth     # Best model checkpoint
```

## Checkpoints and Outputs

- Model checkpoints saved to `synthRAD_checkpoints/Synth_A_to_B.pth`
- Evaluation outputs written as `.mat` files (predictions, labels, loss) during training
- Inference outputs are `.mha` files with CT predictions
