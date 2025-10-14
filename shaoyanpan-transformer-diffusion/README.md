Originally from [this repo](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/blob/main/README.md?plain=1)

# Usage

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

Preprocess SynthRAD into `.mat` files:

```bash
python preprocess_synthrad.py --synthrad_location <path-to-synthrad>
```

This generates normalized MRI/CT `.mat` files under:
- `SynthRAD/imagesTr`
- `SynthRAD/imagesTs`

## Training

Run the main script with custom epochs:

```bash
python main.py --epochs 50
```

## Features added

- tqdm progress bars for training and evaluation

- Epochs configurable via `--epochs`

- MONAI transforms updated with `channel_dim="no_channel"`

- `.mat` loading uses `appendmat=False`

## Checkpoints and Outputs

- Model checkpoints saved to

```
synthRAD_checkpoints/Synth_A_to_B.pth
```

- Evaluation outputs written as `.mat` files (predictions, labels, loss) in the same folder.