import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"  # Can be changed for multi-GPU setups

from lora_train import train

import argparse

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
parser.add_argument('--use_multi_gpu', type=int, required=False, default=0,
                    help='Use multiple GPUs with DistributedDataParallel')
parser.add_argument('--cpu_offload', type=int, required=False, default=0,
                    help='Offload parameters to CPU when not in use')
parser.add_argument('--target_modules', type=str, required=False, 
                    default='all', help='Which modules to apply LoRA to: all, query_proj, key_proj, value_proj, out_proj')

args = parser.parse_args()

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
    use_multi_gpu=args.use_multi_gpu,
    cpu_offload=args.cpu_offload,
    target_modules=args.target_modules
)