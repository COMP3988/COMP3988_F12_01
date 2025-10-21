#!/bin/bash

ROOT=/path/to/dataset    # where you want conditional_DDPM to read from
SRC=/path/to/SynthRAD    # original SynthRAD tree with .../<patient>/{mr.mha,ct.mha}

mkdir -p $ROOT/train/{a,b} $ROOT/val/{a,b} $ROOT/test/{a,b}

# Example: loop your train/val/test patient lists
# For each patient $P:
ln -s "$SRC/<section>/$P/mr.mha" "$ROOT/train/a/${P}.mha"
ln -s "$SRC/<section>/$P/ct.mha" "$ROOT/train/b/${P}.mha"
# repeat for val -> $ROOT/val/{a,b}, test -> $ROOT/test/{a,b}