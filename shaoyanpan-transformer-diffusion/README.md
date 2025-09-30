Originally from [this repo](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/blob/main/README.md?plain=1)

## Usage

### Required packages

Download the required packages using a conda environment

```bash
conda env create -f ./environment.yml
```

### Preprocessing

First, preprocess SynthRAD into data usable by the model

```bash
python preprocess_synthrad.py --synthrad_location <location>
```
