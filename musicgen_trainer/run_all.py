import os
import torch
from train_multi import train_multi
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='Train MusicGen on multiple folders')
    parser.add_argument('--dataset_folders', type=str, required=True, 
                        help='Comma-separated list of dataset folders or directory containing multiple dataset folders')
    parser.add_argument('--model_id', type=str, required=False, default='small')
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--use_wandb', type=int, required=False, default=0)
    parser.add_argument('--no_label', type=int, required=False, default=0)
    parser.add_argument('--tune_text', type=int, required=False, default=0)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
    parser.add_argument('--grad_acc', type=int, required=False, default=2)
    parser.add_argument('--warmup_steps', type=int, required=False, default=16)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--use_cfg', type=int, required=False, default=0)
    parser.add_argument('--devices', type=str, required=False, default=None, 
                        help='Comma-separated list of GPU device indices to use (e.g., "0,1,2")')
    parser.add_argument('--continue_training', type=int, required=False, default=1,
                        help='Continue training from previous checkpoint (1) or start fresh for each folder (0)')
    args = parser.parse_args()

    # Parse devices
    devices = None
    if args.devices:
        devices = [int(device.strip()) for device in args.devices.split(',')]
    else:
        devices = list(range(torch.cuda.device_count()))
    
    # Get dataset folders
    dataset_folders = []
    if ',' in args.dataset_folders:
        # Comma-separated list of folders
        dataset_folders = [folder.strip() for folder in args.dataset_folders.split(',')]
    else:
        # Directory containing multiple dataset folders
        if os.path.isdir(args.dataset_folders):
            dataset_folders = [os.path.join(args.dataset_folders, d) 
                              for d in os.listdir(args.dataset_folders) 
                              if os.path.isdir(os.path.join(args.dataset_folders, d))]
            dataset_folders.sort()  # Sort for deterministic order
        else:
            # Try to use glob pattern
            dataset_folders = glob.glob(args.dataset_folders)
            dataset_folders.sort()
    
    if not dataset_folders:
        print(f"No dataset folders found for: {args.dataset_folders}")
        return
    
    print(f"Found {len(dataset_folders)} dataset folders to process:")
    for folder in dataset_folders:
        print(f"  - {folder}")
    
    # Train on each folder sequentially
    checkpoint = None
    for i, folder in enumerate(dataset_folders):
        print(f"\n[{i+1}/{len(dataset_folders)}] Training on folder: {folder}")
        
        # Create save checkpoint every epoch
        save_step = 1
        
        # Train on this folder
        train_multi(
            dataset_path=folder,
            model_id=args.model_id if i == 0 or not args.continue_training else checkpoint,
            lr=args.lr,
            epochs=args.epochs,
            use_wandb=args.use_wandb,
            save_step=save_step,
            no_label=args.no_label,
            tune_text=args.tune_text,
            weight_decay=args.weight_decay,
            grad_acc=args.grad_acc,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            use_cfg=args.use_cfg,
            devices=devices,
        )
        
        # Update checkpoint path for next iteration
        if args.continue_training:
            checkpoint = "models/lm_final.pt"
            print(f"Next training will continue from checkpoint: {checkpoint}")

if __name__ == "__main__":
    main()