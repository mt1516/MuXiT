import os
# Set memory limits before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Configure memory allocator to limit memory usage and reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

from single_train import single_train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_id', type=str, required=False, default='small')
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--epochs', type=int, required=False, default=100)
parser.add_argument('--use_wandb', type=int, required=False, default=0)
parser.add_argument('--save_step', type=int, required=False, default=None)
parser.add_argument('--no_label', type=int, required=False, default=0)
parser.add_argument('--tune_text', type=int, required=False, default=0)
parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
parser.add_argument('--grad_acc', type=int, required=False, default=2)
parser.add_argument('--warmup_steps', type=int, required=False, default=16)
parser.add_argument('--batch_size', type=int, required=False, default=1)
parser.add_argument('--use_cfg', type=int, required=False, default=0)
parser.add_argument('--use_cpu_offload', type=int, required=False, default=1)
parser.add_argument('--memory_efficient', type=int, required=False, default=1)
parser.add_argument('--use_scaler', type=bool, required=False, default=True)
parser.add_argument('--memory_fraction', type=float, required=False, default=0.95)
args = parser.parse_args()

single_train(
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
    use_cpu_offload=args.use_cpu_offload,
    memory_efficient=args.memory_efficient,
    use_scaler=args.use_scaler,
    memory_fraction=args.memory_fraction,
)
