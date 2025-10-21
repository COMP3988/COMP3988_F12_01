#!/bin/bash

python infer.py \
   --ckpt 200_epoch_ckpt/Synth_A_to_B.pth \
   --input "$1" \
   --outdir "$2" \
   --steps 2 \
   --overlap 0.50 \
   --sw-batch 8 \
   --fp16