import os
# Explicitly set only GPUs 5 and 7 to be visible to PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
import torch
import argparse
from audiocraft.models import MusicGen
from lora_train import train, setup_ddp, apply_lora, AudioDataset
from lora_model_parallel import create_model_parallel_lm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import random
import torchaudio
import math
from torch.nn.parallel import DistributedDataParallel as DDP

def preprocess_audio(audio_path, model, duration: int = 30):
    """Load and preprocess audio."""
    device = model.devices[0]  # Use first device for preprocessing
    
    # Get the sample rate - use the stored attribute or get it from the original model
    sample_rate = model.sample_rate if not hasattr(model, '_model_sample_rate') else model._model_sample_rate
    
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, sample_rate)
    
    # Check if we need to convert mono to stereo for stereo models
    if model.compression_model.channels == 2 and wav.shape[0] == 1:
        # Convert mono to stereo by duplicating the channel
        wav = wav.repeat(2, 1)
        print(f"Converted mono audio to stereo. New shape: {wav.shape}")
    elif model.compression_model.channels == 1 and wav.shape[0] == 2:
        # Convert stereo to mono by averaging channels
        wav = wav.mean(dim=0, keepdim=True)
        print(f"Converted stereo audio to mono. New shape: {wav.shape}")
    
    # Check if audio is long enough
    if wav.shape[1] < sample_rate * duration:
        print(f"Audio too short: {wav.shape[1]/sample_rate:.2f}s < {duration}s")
        return None
    
    # Sample a segment of the audio
    end_sample = int(sample_rate * duration)
    start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
    wav = wav[:, start_sample : start_sample + end_sample]
    
    # Move audio to the appropriate device
    wav = wav.to(device)
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    # Encode the audio
    with torch.no_grad():
        gen_audio = model.compression_model.to(device).encode(wav)
    
    codes, scale = gen_audio
    assert scale is None
    
    return codes

def one_hot_encode(tensor, num_classes=2048):
    """Convert tensor to one-hot encoding."""
    shape = tensor.shape
    device = tensor.device
    one_hot = torch.zeros((shape[0], shape[1], num_classes), device=device, requires_grad=False)

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

def train_model_parallel(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool,
    no_label: bool = False,
    tune_text: bool = False,
    save_step: int = None,
    grad_acc: int = 8,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 3,  # Smaller batch size for model parallelism
    use_cfg: bool = False,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    gpu_ids: list = [0, 1],  # Changed to local device IDs (0,1 when using CUDA_VISIBLE_DEVICES)
    cpu_offload: bool = False,
    target_modules: str = "all",
    fp16: bool = False  # Set to True if you want to use half precision
):
    """Train using model parallelism with LoRA."""
    
    # Check if GPUs are available
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available")
    
    # When using CUDA_VISIBLE_DEVICES, the visible GPUs are mapped to indices starting from 0
    # So even though we're targeting GPUs 5 and 7, inside PyTorch they become 0 and 1
    local_gpu_ids = list(range(len(gpu_ids)))
    print(f"Using logical GPU indices: {local_gpu_ids}")
    
    devices = [torch.device(f'cuda:{i}') for i in local_gpu_ids]
    print(f"Mapped devices: {devices}")
    
    # Set primary device for initial model loading
    primary_device = devices[0]  # This will be cuda:0 when using CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(primary_device)
    
    # Load model on CPU first
    print(f"Loading MusicGen model '{model_id}'...")
    model = MusicGen.get_pretrained(model_id)
    
    # Store the sample_rate before modifying the model
    sample_rate = model.sample_rate
    
    # Explicitly convert model components to float32 to avoid dtype issues
    print("Converting model components to float32 for consistent computation...")
    # Convert compression model parameters to float32
    if hasattr(model.compression_model, 'parameters'):
        for param in model.compression_model.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.float()
    
    # Convert language model parameters to float32
    if hasattr(model.lm, 'parameters'):
        for param in model.lm.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.float()
    
    # Move compression model to the primary device
    model.compression_model = model.compression_model.to(primary_device)
    
    # Create model-parallel version of the language model
    print(f"Creating model-parallel version of the language model...")
    model.lm = create_model_parallel_lm(model.lm, gpu_ids)
    
    # Set up model properties for consistency with original
    model.devices = devices
    # Don't try to set sample_rate directly, store it as a separate attribute
    model._model_sample_rate = sample_rate
    
    # Apply LoRA to the language model
    print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Add LoRA adapters to all eligible layers
    trainable_params_count = 0
    total_params_count = 0
    
    if isinstance(target_modules, str):
        from lora_train import get_lora_target_modules
        target_modules = get_lora_target_modules(model.lm, target_modules)
    
    print(f"Target modules for LoRA: {target_modules}")
    
    # Apply LoRA to each device's layers
    for device_idx, device in enumerate(devices):
        for name, module in model.lm.transformer.device_layers[device_idx].named_modules():
            module_name = name.split(".")[-1]
            if module_name in target_modules and isinstance(module, nn.Linear):
                if hasattr(module, "weight"):
                    # Add to param count
                    total_params_count += module.weight.numel()
                    
                    # Skip if already modified
                    if hasattr(module, "lora_A"):
                        continue
                    
                    # Create LoRA modules on the correct device
                    lora_A = nn.Linear(module.in_features, lora_r, bias=False).to(device)
                    lora_B = nn.Linear(lora_r, module.out_features, bias=False).to(device)
                    
                    # Initialize weights
                    nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                    nn.init.zeros_(lora_B.weight)
                    
                    trainable_params_count += lora_A.weight.numel() + lora_B.weight.numel()
                    
                    # Add dropout
                    lora_dropout_layer = nn.Dropout(p=lora_dropout).to(device)
                    
                    # Store as attributes
                    module.lora_A = lora_A
                    module.lora_B = lora_B
                    module.lora_dropout = lora_dropout_layer
                    module.scaling = lora_alpha / lora_r
                    
                    # Save original forward
                    module.original_forward = module.forward
                    
                    # Create LoRA forward
                    def make_lora_forward(module):
                        original_forward = module.original_forward
                        
                        def lora_forward(x):
                            result = original_forward(x)
                            # Ensure we're on the correct device
                            lora_output = module.lora_B(module.lora_A(module.lora_dropout(x))) * module.scaling
                            return result + lora_output
                        
                        return lora_forward
                    
                    # Replace forward
                    module.forward = make_lora_forward(module)
                    
                    # Make LoRA params trainable only
                    for param in module.lora_A.parameters():
                        param.requires_grad = True
                    for param in module.lora_B.parameters():
                        param.requires_grad = True
                    module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False
    
    # Check linears (output projections) which are on the last device
    if hasattr(model.lm, 'linears'):
        last_device = devices[-1]
        for i, linear in enumerate(model.lm.linears):
            for module_name, module in linear.named_modules():
                if not module_name:  # The linear itself
                    if isinstance(module, nn.Linear) and module_name in target_modules:
                        if hasattr(module, "weight"):
                            # Add to param count
                            total_params_count += module.weight.numel()
                            
                            # Skip if already modified
                            if hasattr(module, "lora_A"):
                                continue
                            
                            # Create LoRA modules on the correct device
                            lora_A = nn.Linear(module.in_features, lora_r, bias=False).to(last_device)
                            lora_B = nn.Linear(lora_r, module.out_features, bias=False).to(last_device)
                            
                            # Initialize weights
                            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                            nn.init.zeros_(lora_B.weight)
                            
                            trainable_params_count += lora_A.weight.numel() + lora_B.weight.numel()
                            
                            # Add dropout
                            lora_dropout_layer = nn.Dropout(p=lora_dropout).to(last_device)
                            
                            # Store as attributes
                            module.lora_A = lora_A
                            module.lora_B = lora_B
                            module.lora_dropout = lora_dropout_layer
                            module.scaling = lora_alpha / lora_r
                            
                            # Save original forward
                            module.original_forward = module.forward
                            
                            # Replace forward
                            module.forward = make_lora_forward(module)
                            
                            # Make LoRA params trainable only
                            for param in module.lora_A.parameters():
                                param.requires_grad = True
                            for param in module.lora_B.parameters():
                                param.requires_grad = True
                            module.weight.requires_grad = False
                            if module.bias is not None:
                                module.bias.requires_grad = False
    
    print(f"Added LoRA modules! Trainable params: {trainable_params_count:,} ({100 * trainable_params_count / total_params_count:.6f}% of total {total_params_count:,})")
    
    # Initialize wandb if requested
    if use_wandb:
        run = wandb.init(project="musicgen-lora-modelparallel")
        run.config.update({
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules
        })

    # Prepare dataset and dataloader
    dataset = AudioDataset(dataset_path, no_label=no_label)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Collect trainable parameters for optimizer
    optimizer_parameters = []
    for device_idx, device in enumerate(devices):
        # Get LoRA parameters from each device
        for _, module in model.lm.transformer.device_layers[device_idx].named_modules():
            if hasattr(module, "lora_A"):
                optimizer_parameters.extend(module.lora_A.parameters())
                optimizer_parameters.extend(module.lora_B.parameters())
    
    # Also get parameters from the linears (output projections)
    if hasattr(model.lm, 'linears'):
        for linear in model.lm.linears:
            for _, module in linear.named_modules():
                if hasattr(module, "lora_A"):
                    optimizer_parameters.extend(module.lora_A.parameters())
                    optimizer_parameters.extend(module.lora_B.parameters())
    
    # Set up optimizer
    optimizer = optim.AdamW(
        optimizer_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    
    # Set up learning rate scheduler
    from transformers import get_scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory for model checkpoints
    save_path = "models/lora_model_parallel/"
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize training state
    save_models = False if save_step is None else True
    current_step = 0

    # Main training loop
    for epoch in range(epochs):
        for batch_idx, (audio, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            all_codes = []
            texts = []

            # Process batch data
            for inner_audio, l in zip(audio, label):
                inner_audio = preprocess_audio(inner_audio, model)
                if inner_audio is None:
                    continue

                if use_cfg:
                    codes = torch.cat([inner_audio, inner_audio], dim=0)
                else:
                    codes = inner_audio

                all_codes.append(codes)
                texts.append(open(l, "r").read().strip())

            # Skip batch if no valid audio
            if len(all_codes) == 0:
                continue
                
            # Prepare conditions - need to be on first device
            attributes, _ = model._prepare_tokens_and_attributes(texts, None)
            conditions = attributes
            
            # Apply classifier free guidance if needed
            if use_cfg:
                from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions
                
            # Tokenize and condition the model - ensure tensors are on the right device
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions

            codes = torch.cat(all_codes, dim=0)  # Will be moved to device in compute_predictions

            # Forward pass through the model
            lm_output = model.lm.compute_predictions(
                codes=codes, conditions=[], condition_tensors=condition_tensors
            )

            # Extract predictions and targets - all on last device now
            last_device = devices[-1]
            codes_target = codes[0].to(last_device)
            logits = lm_output.logits[0]
            mask = lm_output.mask[0]

            # Debug shapes to understand the mismatch
            # print(f"Original shapes - logits: {logits.shape}, mask: {mask.shape}, codes_target: {codes_target.shape}")
            
            # Convert to one-hot encoding
            codes_target = one_hot_encode(codes_target, num_classes=2048)
            
            # Ensure mask and logits have compatible shapes before indexing
            # Flatten logits and codes while ensuring alignment with mask
            flattened_logits = logits.reshape(-1, 2048)
            flattened_codes = codes_target.reshape(-1, 2048)
            flattened_mask = mask.reshape(-1)
            
            # print(f"Flattened shapes - logits: {flattened_logits.shape}, mask: {flattened_mask.shape}")
            
            # Make sure mask isn't longer than the flattened tensors
            mask_length = min(flattened_mask.shape[0], flattened_logits.shape[0])
            flattened_mask = flattened_mask[:mask_length]
            flattened_logits = flattened_logits[:mask_length]
            flattened_codes = flattened_codes[:mask_length]
            
            # Apply mask by selecting only valid positions
            valid_indices = flattened_mask.nonzero().squeeze()
            if valid_indices.numel() > 0:
                masked_logits = flattened_logits[valid_indices]
                masked_codes = flattened_codes[valid_indices]
                
                # Compute loss only on valid positions
                loss = criterion(masked_logits, masked_codes)
            else:
                print("Warning: No valid positions found in mask!")
                loss = torch.tensor(0.0, device=last_device, requires_grad=True)

            # Update step counter
            current_step += 1 / grad_acc

            # Backward pass
            loss.backward()

            # Skip gradient update if not at accumulation boundary
            if batch_idx % grad_acc != grad_acc - 1:
                continue
                
            # Calculate gradient norm for logging
            total_norm = 0
            for p in optimizer_parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Clip gradients to prevent explosions
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, 0.5)

            # Optimizer step
            optimizer.step()
                
            # Update learning rate
            scheduler.step()

            # Log metrics
            if use_wandb:
                run.log({
                    "loss": loss.item(),
                    "total_norm": total_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            # Print progress
            print(
                f"Epoch: {epoch}/{epochs}, Batch: {batch_idx}/{len(train_dataloader)}, "
                f"Loss: {loss.item():.6f}, Grad Norm: {total_norm:.6f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save model checkpoint if requested
            if save_models and int(current_step) % save_step == 0:
                save_checkpoint(model, f"{save_path}/musicgen_lora_{int(current_step)}.pt", devices)
                print(f"Saved checkpoint at step {int(current_step)}")

        # Save checkpoint after each epoch
        save_checkpoint(model, f"{save_path}/musicgen_lora_epoch_{epoch+1}.pt", devices)
        print(f"Saved checkpoint after epoch {epoch+1}")

    # Save final model
    save_checkpoint(model, f"{save_path}/musicgen_lora_final.pt", devices)
    print("Training complete! Final model saved.")


def save_checkpoint(model, path, devices):
    """Save a LoRA checkpoint with only the adapter weights."""
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save LoRA weights only
    lora_state_dict = {}
    
    # Collect weights from all devices
    for device_idx, device in enumerate(devices):
        for name, module in model.lm.transformer.device_layers[device_idx].named_modules():
            if hasattr(module, "lora_A"):
                module_name = f"transformer.device_layers.{device_idx}.{name}"
                lora_state_dict[f"{module_name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
                lora_state_dict[f"{module_name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    
    # Also save linear weights if they have LoRA
    if hasattr(model.lm, 'linears'):
        for i, linear in enumerate(model.lm.linears):
            for name, module in linear.named_modules():
                if hasattr(module, "lora_A"):
                    module_name = f"linears.{i}.{name}"
                    lora_state_dict[f"{module_name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
                    lora_state_dict[f"{module_name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    
    torch.save({"lora_state_dict": lora_state_dict}, path)
    print(f"Saved LoRA weights to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train MusicGen with model-parallel LoRA")
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
    parser.add_argument('--batch_size', type=int, required=False, default=3)
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
    parser.add_argument('--gpu_ids', type=str, required=False, default='5,7',
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--fp16', type=int, required=False, default=0,
                        help='Use half precision for training')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    
    # Run training with model parallelism
    train_model_parallel(
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
        gpu_ids=gpu_ids,
        cpu_offload=args.cpu_offload,
        target_modules=args.target_modules,
        fp16=args.fp16
    )


if __name__ == "__main__":
    main()