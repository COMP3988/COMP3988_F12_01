Originally from [this repo](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/blob/main/README.md?plain=1)

## Usage

### Installing Mamba

If you don't already have Mamba installed, you can add it to your base conda environment:

```bash
conda install -n base -c conda-forge mamba
```

Alternatively, you can install the lightweight standalone **micromamba** (no conda needed) by running:

```bash
curl micro.mamba.pm/install.sh | bash
```

After installation, restart your shell or source your `~/.bashrc` or `~/.zshrc`

### Required packages

It is recommended to use [mamba](https://github.com/mamba-org/mamba) instead of conda, as it resolves dependencies and installs packages much faster while remaining fully compatible.

```bash
conda env create -f ./environment.yml
```

### Preprocessing

First, preprocess SynthRAD into data usable by the model

```bash
python preprocess_synthrad.py --synthrad_location <location>
```
