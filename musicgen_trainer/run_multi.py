import os
import torch
import argparse
from train_multi import train_multi
import gc
import sys  # Add for debugging info
from torch.multiprocessing import set_start_method

try:
    # Set multiprocessing start method to spawn for proper CUDA behavior
    set_start_method('spawn', force=True)
except RuntimeError:
    # Already set
    pass

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
parser.add_argument('--batch_size', type=int, required=False, default=4)
parser.add_argument('--per_device_batch_size', type=int, required=False, default=None,
                    help='Batch size per GPU (overrides batch_size if provided)')
parser.add_argument('--use_cfg', type=int, required=False, default=0)
parser.add_argument('--devices', type=str, required=False, default=None, 
                    help='Comma-separated GPU indices (e.g. "1,2,3,4,5,6,7")')
parser.add_argument('--gradient_checkpointing', type=int, required=False, default=0,
                    help='Use gradient checkpointing to reduce memory usage')
parser.add_argument('--pin_memory', type=int, required=False, default=1,
                    help='Use pinned memory for data loading')
parser.add_argument('--memory_efficient_attention', type=int, required=False, default=1,
                    help='Use memory efficient attention implementation')
parser.add_argument('--mixed_precision', type=str, required=False, default='bf16', 
                    choices=[None, 'fp16', 'bf16'], help='Use mixed precision training')
parser.add_argument('--cpu_offload', type=int, required=False, default=1,
                    help='Offload optimizer states to CPU to save GPU memory')
args = parser.parse_args()

if args.devices:
    devices = [int(d.strip()) for d in args.devices.split(',')]
else:
    # Check free memory on all GPUs
    if torch.cuda.is_available():
        free_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            gc.collect()  # Clean up Python objects
            
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free_mem = total_mem - allocated
            free_memory.append((i, free_mem, total_mem))
            print(f"GPU {i}: {props.name}, {free_mem:.2f}GB free of {total_mem:.2f}GB")
            
        # Sort GPUs by free memory (descending)
        free_memory.sort(key=lambda x: x[1], reverse=True)
        
        # Select GPUs with more than 5GB free memory
        devices = [idx for idx, free, total in free_memory if free > 5.0]
        
        if not devices:
            print("WARNING: No GPUs with >5GB free memory found, using just the first GPU")
            devices = [0]
        else:
            print(f"Selected {len(devices)} GPUs with >5GB free memory: {devices}")
    else:
        devices = []

# Make sure to clear CUDA cache before doing anything
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Print CUDA diagnostic information
print("\n---- GPU Diagnostics ----")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Check environment variables
print("\n---- Environment Variables ----")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'Not set')}")

if len(devices) > 0:
    primary_device = devices[0]
    print(f"\nUsing GPUs:")
    print(f"  - Model (primary): cuda:{primary_device}")
    print(f"  - Data preprocessing: {devices}")
    
    # Force primary CUDA device
    torch.cuda.set_device(primary_device)
    print(f"Set current CUDA device to: cuda:{primary_device}")
    
    # Set visible devices clearly
    visible_devices = ",".join(map(str, devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ensure consistent device order
    
    # MODIFIED: Don't set default tensor type which can cause compatibility issues
    # Instead just verify CUDA works
    if torch.cuda.is_available():
        print(f"CUDA confirmed available with {torch.cuda.device_count()} devices")
        # Create a test tensor to verify CUDA works
        try:
            test_tensor = torch.zeros(1, device=f"cuda:{primary_device}")
            print(f"Test tensor created on {test_tensor.device}")
            del test_tensor
        except Exception as e:
            print(f"Error creating CUDA tensor: {e}")
    else:
        print("WARNING: CUDA not available despite GPU selection!")
else:
    print("No GPUs available, using CPU only")

# Calculate effective batch size
effective_batch_size = args.batch_size
per_device_batch_size = args.per_device_batch_size

if per_device_batch_size is not None:
    print(f"Using per-device batch size: {per_device_batch_size}")
    effective_batch_size = per_device_batch_size * len(devices) if len(devices) > 0 else per_device_batch_size
    print(f"Effective total batch size: {effective_batch_size}")
else:
    if len(devices) > 0:
        per_device_batch_size = max(1, args.batch_size // len(devices))
    else:
        per_device_batch_size = args.batch_size
    print(f"Calculated per-device batch size: {per_device_batch_size}")
    print(f"Total batch size: {args.batch_size}")

# Print memory management configuration
print(f"Memory management:")
print(f"  - Gradient accumulation steps: {args.grad_acc}")
print(f"  - Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
print(f"  - Memory efficient attention: {'Enabled' if args.memory_efficient_attention else 'Disabled'}")
print(f"  - Mixed precision: {args.mixed_precision if args.mixed_precision else 'Disabled'}")
print(f"  - CPU offloading: {'Enabled' if args.cpu_offload else 'Disabled'}")

# Explicitly restore CUDA visibility in both variables to avoid any confusion
original_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
temp_visible_devices = original_devices  # Save for restoration

# Temporarily hide devices during model loading only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Debug flags to help with CUDA issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors more debuggable
print(f"\nStarting training process with GPU debugging enabled")

try:
    # Set environment variable for PyTorch distributed
    if len(devices) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(len(devices))
        print(f"Configured environment for distributed training with {len(devices)} GPUs")
    
    # Force loading on CPU, but enable GPU for training
    print("Starting training with explicit GPU acceleration...")
    train_multi(
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
        batch_size=per_device_batch_size,  # Pass per-device batch size
        use_cfg=args.use_cfg,
        devices=devices,
        gradient_checkpointing=args.gradient_checkpointing,
        memory_efficient_attention=args.memory_efficient_attention,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        pin_memory=args.pin_memory,
        force_cpu_loading=True,  # Force CPU loading to avoid OOM
        restore_devices=temp_visible_devices,  # Pass the original devices to be restored
        # Additional GPU forcing parameters
        force_gpu=True,
        allow_tf32=True,  # Enable TF32 precision for better performance on Ampere+
        # Add explicit memory threshold to avoid OOM
        # max_memory_threshold=5.0,  # GB per GPU
        # Move more tensor operations to CPU to avoid GPU OOM
        # force_cpu_preprocessing=True,
        multi_gpu=len(devices) > 1,  # Enable multi GPU only if we have multiple usable GPUs
        # use_distributed=len(devices) > 1,  # Use DistributedDataParallel for better performance
    )
finally:
    # Cleanup
    if 'CUDA_LAUNCH_BLOCKING' in os.environ:
        del os.environ['CUDA_LAUNCH_BLOCKING']
    os.environ['CUDA_VISIBLE_DEVICES'] = original_devices
    print(f"Training completed, restored CUDA visibility to: {original_devices}")