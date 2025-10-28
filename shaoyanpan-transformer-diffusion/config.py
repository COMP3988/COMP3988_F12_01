"""
Configuration file containing all hyperparameters for the transformer-diffusion model.
This file centralizes all model, training, and inference parameters to ensure consistency
between training (main.py) and inference (infer_single.py) scripts.
"""

import torch

# =============================================================================
# DATA LOADER HYPERPARAMETERS
# =============================================================================
BATCH_SIZE_TRAIN = 4 * 1
IMG_SIZE = (256, 256, 128)
PATCH_SIZE = (64, 64, 2)
SPACING = (2, 2, 2)
PATCH_NUM = 1
CHANNELS = 1

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
NUM_CHANNELS = 64
ATTENTION_RESOLUTIONS = "32,16,8"
CHANNEL_MULT = (1, 2, 3, 4)
NUM_HEADS = [4, 4, 8, 16]
WINDOW_SIZE = [[4, 4, 2], [4, 4, 2], [4, 4, 2], [4, 4, 2]]
NUM_RES_BLOCKS = [1, 1, 1, 1]
SAMPLE_KERNEL = ([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1])

# Derived parameters
ATTENTION_DS = [int(x) for x in ATTENTION_RESOLUTIONS.split(",")]
CLASS_COND = False
USE_SCALE_SHIFT_NORM = True
RESBLOCK_UPDOWN = False
DROPOUT = 0

# =============================================================================
# DIFFUSION HYPERPARAMETERS
# =============================================================================
DIFFUSION_STEPS = 1000
LEARN_SIGMA = True
SIGMA_SMALL = False
NOISE_SCHEDULE = "linear"
USE_KL = False
PREDICT_XSTART = True
RESCALE_TIMESTEPS = True
RESCALE_LEARNED_SIGMAS = True

# Training-specific diffusion parameters
TIMESTEP_RESPACING = [50]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
USE_CHECKPOINT = False
USE_FP16 = False
USE_NEW_ATTENTION_ORDER = False

# =============================================================================
# INFERENCE HYPERPARAMETERS
# =============================================================================
# Default inference parameters
DEFAULT_INFERENCE_STEPS = 2
DEFAULT_OVERLAP = 0.50
DEFAULT_SW_BATCH = 8

# =============================================================================
# EVALUATION HYPERPARAMETERS
# =============================================================================
EVAL_STEPS = 10
EVAL_OVERLAP = 0.0
EVAL_SW_BATCH = 32

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# MODEL CONFIGURATION FUNCTIONS
# =============================================================================
def get_model_config():
    """Get the complete model configuration dictionary."""
    return {
        'image_size': PATCH_SIZE,
        'in_channels': 2,
        'model_channels': NUM_CHANNELS,
        'out_channels': 2,
        'dims': 3,
        'sample_kernel': SAMPLE_KERNEL,
        'num_res_blocks': NUM_RES_BLOCKS,
        'attention_resolutions': tuple(ATTENTION_DS),
        'dropout': DROPOUT,
        'channel_mult': CHANNEL_MULT,
        'num_classes': None,
        'use_checkpoint': USE_CHECKPOINT,
        'use_fp16': USE_FP16,
        'num_heads': NUM_HEADS,
        'window_size': WINDOW_SIZE,
        'num_head_channels': 64,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': USE_SCALE_SHIFT_NORM,
        'resblock_updown': RESBLOCK_UPDOWN,
        'use_new_attention_order': USE_NEW_ATTENTION_ORDER,
    }

def get_diffusion_config():
    """Get the complete diffusion configuration dictionary."""
    return {
        'steps': DIFFUSION_STEPS,
        'learn_sigma': LEARN_SIGMA,
        'sigma_small': SIGMA_SMALL,
        'noise_schedule': NOISE_SCHEDULE,
        'use_kl': USE_KL,
        'predict_xstart': PREDICT_XSTART,
        'rescale_timesteps': RESCALE_TIMESTEPS,
        'rescale_learned_sigmas': RESCALE_LEARNED_SIGMAS,
    }

def get_training_diffusion_config():
    """Get the diffusion configuration for training (includes timestep_respacing)."""
    config = get_diffusion_config()
    config['timestep_respacing'] = TIMESTEP_RESPACING
    return config
