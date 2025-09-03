~~hacked together~~ adapted from https://github.com/milesial/Pytorch-UNet/

# Usage

### Installing dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Training

To train on the MRI→CT dataset:

```bash
python3 unet/train.py \
  --data-root ./synthRAD2025_Task1_Train/Task1 \
  --target ct \
  --epochs 1 \
  --batch-size 1 \
  --scale 1.0 \
  --samples-per-section 10
```

Checkpoints will be saved under `./checkpoints/`.

### Prediction

To generate predictions on an .mha volume:

```bash
python predict.py \
  --model checkpoints/checkpoint_epoch20.pth \
  --input synthRAD2025_Task1_Train/Task1/AB/1ABA005/mr.mha \
  --output out.mha
```

TODO: might have not updated predict.py yet

This will output the predicted volume to `out.mha`.