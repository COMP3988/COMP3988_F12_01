# Transformer-Diffusion Model for MRI→CT Synthesis

A 3D transformer-based denoising diffusion model for synthetic CT generation from MRI scans. This workflow covers the complete pipeline from environment setup through preprocessing, training, inference, and post-processing visualization.

Originally from [this repo](https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model)

## Environment Setup

Target environment: **Python 3.10** with **CUDA 12.1** (tested on NVIDIA L4 GPU).
We recommend using **mamba** or **micromamba** for dependency resolution.

### Install micromamba (optional)

```bash
curl micro.mamba.pm/install.sh | bash
```

Restart your shell or source your `~/.bashrc` / `~/.zshrc`.

### Create environment

```bash
mamba create -n synthrad-l4 python=3.10
mamba activate synthrad-l4
```

### Install PyTorch with CUDA

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install supporting packages

```bash
mamba install -c conda-forge monai einops timm natsort scikit-image scipy tqdm matplotlib napari
```

If you hit issues with `pkg_resources` deprecation:

```bash
pip install "setuptools<81"
```

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Complete Workflow

### 1. Preprocessing

Preprocess your SynthRAD dataset (`.mha` files) into normalized `.npz` files for training:

```bash
python preprocess_synthrad.py --synthrad_location <path-to-synthrad>
```

This generates normalized MRI/CT `.npz` files under:
- `SynthRAD/imagesTr` (training, 70%)
- `SynthRAD/imagesVal` (validation, 15%)
- `SynthRAD/imagesTs` (testing, 15%)

Each `.npz` file contains:
- `image`: MRI volume normalized to [-1, 1]
- `label`: CT volume normalized to [-1, 1]

**Note**: `.npz` files are only used during the preprocessing and training phases. For inference, you work directly with `.mha` files.

### 2. Training

Train the diffusion model:

```bash
python main.py --epochs 50
```

#### Training Details

- Model automatically saves checkpoints to `synthRAD_checkpoints/Synth_A_to_B.pth`
- Training loss: MSE-based diffusion loss (reported per epoch)
- Evaluation loss: L1 Loss (Mean Absolute Error) between predicted and ground truth CT (evaluated every 5 epochs)
- Best model is saved based on lowest evaluation L1 loss
- Progress bars show training loss updates during each epoch

#### Training Parameters

- Batch size: 4
- Learning rate: 2e-5 (AdamW optimizer)
- Device: cuda:0 (automatically uses GPU if available)
- Checkpoint directory: `synthRAD_checkpoints/`

### 3. Inference

#### Single File Inference

Process a single MRI volume (.mha) to generate synthetic CT (.mha):

```bash
python infer_single.py \
  --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
  --input path/to/input_mri.mha \
  --output output_ct.mha
```

#### Batch Directory Inference

Process all MRI files (.mha/.mhd) in a directory:

```bash
python infer_directory.py \
  --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
  --inputdir path/to/mri_files \
  --outputdir outputs_mha
```

**Important**: The raw synthetic CT output from inference may appear significantly brighter than ground truth CT volumes. For proper visualization and quantitative evaluation, use `view.py` (see Step 4 below) to histogram-match the synthetic CT to ground truth.

#### Performance Optimization

Default parameters are balanced for quality and speed. For faster inference on L4 GPU:

```bash
python infer_single.py \
  --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
  --input your_input.mha \
  --output output.mha \
  --sw-batch 1024 \
  --fp16
```

- `--sw-batch 1024`: Increases sliding window batch size for faster processing
- `--fp16`: Enables mixed precision inference for memory reduction and potential speedup

#### Required Arguments

- `--ckpt`: Path to trained model checkpoint (.pth file)
- `--input`: Input MRI file (.mha or .mhd format; .npz supported for backwards compatibility)
- `--output`: Output path for synthetic CT (.mha file)

#### Optional Arguments

- `--steps`: Number of diffusion steps (default: 2)
  - Higher values (e.g., 10-50) yield better quality but significantly slower inference
  - Default is 2 for balancing quality and speed (~2-3 min per volume on L4 GPU)
  - Trade-off: quality vs time (2 steps ≈ 2-3 min, 10 steps ≈ 10-15 min)
- `--overlap`: Sliding-window overlap ratio [0,1) (default: 0.5)
- `--sw-batch`: Sliding-window batch size (default: 512)
- `--fp16`: Enable mixed precision inference for memory reduction and speedup
- `--device`: Device to use (default: cuda:0)
- `--mask`: Optional mask file for focused processing (default: "infer")

### 4. Post-Processing and Visualization with `view.py`

The `view.py` script performs histogram matching to align the synthetic CT intensity distribution with ground truth CT, then visualizes the results in Napari.

#### Usage

```bash
python view.py \
  --gt path/to/ground_truth_ct.mha \
  --sct path/to/synthetic_ct.mha \
  --output matched_sct.mha
```

#### Purpose

1. **Histogram Matching**: Aligns the intensity distribution of synthetic CT to match ground truth CT using `skimage.exposure.match_histograms`
2. **Visual Comparison**: Opens Napari viewer showing:
   - Ground truth CT
   - Original synthetic CT
   - Matched synthetic CT (before fade)
   - Final output (matched + faded)
3. **Bright Spot Removal**: Applies optional high-HU darkening to remove artifacts

#### Optional Parameters

- `--window MIN MAX`: Display window in HU units (default: -200 300)
- `--fade-knee`: HU value where fading begins (default: 275.0)
- `--fade-width`: Transition width in HU for the fade (default: 80.0)
- `--fade-target`: HU target for very bright regions (default: -1000.0)
- `--no-fade`: Disable high-HU darkening after histogram match

## File Structure

```
shaoyanpan-transformer-diffusion/
├── main.py                    # Training script
├── infer_single.py           # Single file inference
├── infer_directory.py        # Batch directory inference
├── view.py                   # Histogram matching and visualization
├── preprocess_synthrad.py   # SynthRAD preprocessing
├── diffusion/                # Diffusion model components
├── network/                  # Transformer network architecture
├── SynthRAD/                # Preprocessed data (created by preprocessing)
│   ├── imagesTr/           # Training .npz files
│   ├── imagesVal/          # Validation .npz files
│   └── imagesTs/           # Testing .npz files
└── synthRAD_checkpoints/    # Model checkpoints
    └── Synth_A_to_B.pth     # Best model checkpoint
```

## Output Handling Notes

### Brightness Mismatch Without `view.py`

**Important**: Without using `view.py` for histogram matching, the synthetic CT volumes will appear significantly brighter than ground truth CT volumes. This is due to the intensity distribution differences between the raw diffusion model output and real CT scans. Always use `view.py` to histogram-match synthetic CT to ground truth for proper visualization and fair quantitative comparisons.

### Output Inversion

During inference, the model outputs CT values that are inverted relative to the typical HU scale. This is expected behavior due to the diffusion process and the normalized input/output ranges. The values are in the normalized range [-1, 1] and may appear inverted when viewed directly.

### Bright Spot Removal in `view.py`

The `view.py` script applies a "fade" operation to remove bright artifacts that may appear in the synthetic CT output. The fade operation:

1. Identifies high-intensity regions above a knee value (default: 275 HU)
2. Smoothly transitions these values toward air-equivalent intensity (default: -1000 HU)
3. Uses a smoothstep function for a gradual, non-oscillatory transition

This helps remove unrealistic bright spots that may be generated during the diffusion process, particularly at bone-air interfaces or in regions with high gradient artifacts.

The fade is applied after histogram matching, so the final output in `view.py` shows both the intensity-matched synthetic CT and the additional artifact suppression. Use the `--no-fade` flag if you want to see the matched output without the bright spot removal.
