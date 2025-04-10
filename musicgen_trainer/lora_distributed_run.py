import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from lora_train import train, setup_ddp


def run_distributed(rank, world_size, args):
    """Run training on a specific GPU rank."""
    # Set device for this process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Only display on first GPU
    is_main = rank == 0
    
    train(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb if is_main else False,  # Only log on main process
        save_step=args.save_step,
        no_label=args.no_label,
        tune_text=args.tune_text,
        weight_decay=args.weight_decay,
        grad_acc=args.grad_acc,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_multi_gpu=True,  # Already in distributed mode
        cpu_offload=args.cpu_offload,
        target_modules=args.target_modules
    )
    
    # Cleanup
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=False, default='small')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--use_wandb', type=int, required=False, default=0)
    parser.add_argument('--save_step', type=int, required=False, default=None)
    parser.add_argument('--no_label', type=int, required=False, default=0)
    parser.add_argument('--tune_text', type=int, required=False, default=0)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
    parser.add_argument('--grad_acc', type=int, required=False, default=2)
    parser.add_argument('--warmup_steps', type=int, required=False, default=16)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--use_cfg', type=int, required=False, default=0)

    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, required=False, default=16, 
                        help='LoRA rank parameter - lower means fewer parameters')
    parser.add_argument('--lora_alpha', type=float, required=False, default=32.0,
                        help='LoRA alpha parameter - scaling factor')
    parser.add_argument('--lora_dropout', type=float, required=False, default=0.1,
                        help='LoRA dropout probability')
    parser.add_argument('--cpu_offload', type=int, required=False, default=0,
                        help='Offload parameters to CPU when not in use')
    parser.add_argument('--target_modules', type=str, required=False, 
                        default='all', help='Which modules to apply LoRA to')
    
    args = parser.parse_args()
    
    # Check available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Only 1 GPU detected. Running in single GPU mode.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        train(
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            lr=args.lr,
            epochs=args.epochs,
            use_wandb=args.use_wandb,
            save_step=args.save_step,
            no_label=args.no_label,
            tune_text=args.tune_text,
            weight_decay=args.weight_decay,
            grad_acc=args.grad_acc,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            use_cfg=args.use_cfg,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_multi_gpu=False,
            cpu_offload=args.cpu_offload,
            target_modules=args.target_modules
        )
    else:
        print(f"Found {world_size} GPUs. Running distributed training.")
        # Remove any specific GPU selection as we're managing it through distribution
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Launch a process for each GPU
        mp.spawn(
            run_distributed,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()