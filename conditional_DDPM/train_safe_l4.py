#!/usr/bin/env python3
"""
Safe training configuration for L4 GPU - prevents CUDA OOM
"""
import os
import time
import datetime
import sys
import argparse
import torch

from Diffusion_condition import GaussianDiffusionTrainer_cond
from Model_condition import UNet
from datasets import ImageDataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Safe L4 GPU training')
    parser.add_argument('--dataset_path', type=str, default='../synthRAD2025_Task2_Train/Task2')
    parser.add_argument('--out_name', type=str, default='synthrad_safe_l4')
    parser.add_argument('--batch_size', type=int, default=2)  # Very conservative
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=128)  # Smaller images
    parser.add_argument('--ch', type=int, default=64)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--attn', nargs='+', type=int, default=[1])
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_freq', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Create output directory
    save_weight_dir = f"./Checkpoints/{args.out_name}"
    os.makedirs(save_weight_dir, exist_ok=True)

    # Create dataset and dataloader with conservative settings
    print("Loading dataset...")
    train_dataloader = DataLoader(
        ImageDataset(args.dataset_path, transforms_=False, unaligned=False,
                   image_size=(args.image_size, args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Reduce to 0 to avoid memory issues
        pin_memory=False  # Disable pin_memory to reduce memory usage
    )

    print(f"Dataset loaded with {len(train_dataloader)} batches")

    # Initialize model
    print("Initializing model...")
    net_model = UNet(args.T, args.ch, args.ch_mult, args.attn,
                    args.num_res_blocks, args.dropout).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in net_model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Check GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Initialize optimizer and trainer
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    trainer = GaussianDiffusionTrainer_cond(net_model, args.beta_1, args.beta_T, args.T).to(device)

    # Training loop
    print("Starting training...")
    prev_time = time.time()

    # Track best model
    best_loss = float('inf')
    best_epoch = 0


    for epoch in range(args.n_epochs):
        epoch = epoch + 1

        epoch_loss = 0
        num_batches = 0

        net_model.train()
        for i, batch in enumerate(train_dataloader):
            try:

                optimizer.zero_grad(set_to_none=True)

                ct = batch["a"].to(device)
                cbct = batch["b"].to(device)
                x_0 = torch.cat((ct, cbct), 1)

                loss = trainer(x_0)  # Already returns mean loss
                epoch_loss += loss.item()
                num_batches += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
                optimizer.step()

                # logging
                # if (i + 1) % 5 == 0:
                    # print(f"... Loss(mean): {loss.item():.6f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ CUDA out of memory at epoch {epoch}, batch {i+1}")
                    print("Try reducing batch_size or image_size further")
                    return
                else:
                    raise e

        avg_loss = epoch_loss / num_batches

        # Check if this is the best model so far
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch
            print(f"🎉 New best model! Loss: {avg_loss:.6f} (Epoch {epoch})")

        # Calculate time estimates
        time_duration = datetime.timedelta(seconds=(time.time() - prev_time))
        epoch_left = args.n_epochs - epoch
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - prev_time))
        prev_time = time.time()

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(save_weight_dir, f'ckpt_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_model_path = os.path.join(save_weight_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, best_model_path)
            print(f"Best model saved: {best_model_path}")

        # Print epoch summary
        gpu_mem = torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
        print(f"[Epoch {epoch}/{args.n_epochs}] [ETA: {time_left}] [Duration: {time_duration}] [Loss: {avg_loss:.6f}] [Best: {best_loss:.6f} @ Epoch {best_epoch}] [GPU: {gpu_mem:.2f}GB]")

    # Save final model
    final_path = os.path.join(save_weight_dir, 'final_model.pt')
    torch.save({
        'epoch': args.n_epochs,
        'model_state_dict': net_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    print(f"Final model saved: {final_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: Epoch {best_epoch} with loss {best_loss:.6f}")
    print("="*60)

if __name__ == "__main__":
    main()
