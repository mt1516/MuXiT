import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import wandb
import os
import time
import copy
import math
from typing import List, Dict, Any, Optional, Union
import numpy as np

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

# Import for LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
import torch.distributed as dist

from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout


class AudioDataset(Dataset):
    def __init__(self, data_dir, no_label=False):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == ".wav":
                if no_label:
                    self.data_map.append({"audio": os.path.join(data_dir, d)})
                    continue
                if os.path.exists(os.path.join(data_dir, name + ".txt")):
                    self.data_map.append(
                        {
                            "audio": os.path.join(data_dir, d),
                            "label": os.path.join(data_dir, name + ".txt"),
                        }
                    )
                else:
                    raise ValueError(f"No label file for {name}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data.get("label", "")

        return audio, label


def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    
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
    if wav.shape[1] < model.sample_rate * duration:
        print(f"Audio too short: {wav.shape[1]/model.sample_rate:.2f}s < {duration}s")
        return None
    
    # Sample a segment of the audio
    end_sample = int(model.sample_rate * duration)
    start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
    wav = wav[:, start_sample : start_sample + end_sample]
    
    # Get device from language model
    device = next(model.lm.parameters()).device
    wav = wav.to(device)
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    # Encode the audio
    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)
    
    codes, scale = gen_audio
    assert scale is None
    
    return codes


def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

    return result


def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes), device=tensor.device, requires_grad=False)

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot


def setup_ddp(rank, init_process_group=True):
    """Initialize the distributed training environment."""
    if init_process_group:
        dist.init_process_group("nccl", rank=rank, world_size=torch.cuda.device_count())
    torch.cuda.set_device(rank)


def get_lora_target_modules(model, target_modules="all"):
    """Get the target modules for LoRA adaptation based on the selection."""
    if target_modules == "all":
        # Find all linear layers in the transformer
        target_mods = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "transformer" in name:
                target_mods.add(name.split(".")[-1])
        return list(target_mods)
    elif target_modules == "attention":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        # Custom list of module names
        return target_modules.split(",")


def apply_lora(
    model, 
    lora_r: int = 16, 
    lora_alpha: float = 32.0, 
    lora_dropout: float = 0.1,
    target_modules: Union[str, List[str]] = "all"
):
    """Apply LoRA adapters to the model."""
    
    if isinstance(target_modules, str):
        target_modules = get_lora_target_modules(model, target_modules)
    
    # Instead of cloning the entire state dict, just keep track of target modules
    # This saves GPU memory by avoiding a full model state copy
    model._lora_target_modules = target_modules
    
    print(f"Target modules for LoRA: {target_modules}")
    
    # Config for LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",  # Using string to avoid TaskType dependency
    )
    
    # Custom LoRA application - manually apply to each module
    trainable_params_count = 0
    total_params_count = 0
    
    # Apply LoRA manually to avoid the prepare_inputs_for_generation error
    from peft.tuners.lora import LoraLayer
    
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]
        if module_name in target_modules and isinstance(module, nn.Linear):
            # Create LoRA modules
            if hasattr(module, "weight"):
                device = module.weight.device  # Get the device of the module
                total_params_count += module.weight.numel()
                
                # Check if already modified - skip if already done
                if hasattr(module, "lora_A"):
                    continue
                
                # Add LoRA modules and ensure they're on the same device as the module
                lora_A = nn.Linear(module.in_features, lora_r, bias=False).to(device)
                lora_B = nn.Linear(lora_r, module.out_features, bias=False).to(device)
                
                # Initialize weights - similar to what PEFT does
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
                
                # Create new forward with LoRA that handles device properly
                def make_lora_forward(module):
                    original_forward = module.original_forward
                    
                    def lora_forward(x):
                        # Ensure we're operating on the correct device
                        device = x.device
                        result = original_forward(x)
                        
                        # Move LoRA components to the input's device if needed
                        if module.lora_A.weight.device != device:
                            module.lora_A = module.lora_A.to(device)
                            module.lora_B = module.lora_B.to(device)
                            module.lora_dropout = module.lora_dropout.to(device)
                            
                        lora_output = module.lora_B(module.lora_A(module.lora_dropout(x))) * module.scaling
                        return result + lora_output
                        
                    return lora_forward
                
                # Replace forward method
                module.forward = make_lora_forward(module)
                
                # Mark parameters
                for param in module.lora_A.parameters():
                    param.requires_grad = True
                for param in module.lora_B.parameters():
                    param.requires_grad = True
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
    
    # Freeze all parameters except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False
    
    print(f"Added LoRA modules! Trainable params: {trainable_params_count:,} ({100 * trainable_params_count / total_params_count:.6f}% of total {total_params_count:,})")
    
    return model


def train(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool,
    no_label: bool = False,
    tune_text: bool = False,
    save_step: int = None,
    grad_acc: int = 8,
    use_scaler: bool = False,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 10,
    use_cfg: bool = False,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    use_multi_gpu: bool = False,
    cpu_offload: bool = False,
    target_modules: str = "all",
    init_process_group: bool = True
):
    # Set up distributed training if requested
    is_distributed = use_multi_gpu and torch.cuda.device_count() > 1
    rank = 0
    
    if is_distributed:
        # Get local rank if in distributed mode
        if dist.is_initialized():
            rank = dist.get_rank()
        setup_ddp(rank, init_process_group=init_process_group)
        print(f"Process {rank} using GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(rank)})")
    
    # Initialize wandb if requested
    if use_wandb and (not is_distributed or rank == 0):
        run = wandb.init(project="musicgen-lora")
        run.config.update({
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules
        })

    # Load model and move it to the appropriate device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Process {rank} loading model on {device}")
    
    # Use a memory-efficient approach to load the model on the specific GPU
    with torch.cuda.device(device):
        # Clear any existing cache to make room for the model
        torch.cuda.empty_cache()
        
        # Load model
        model = MusicGen.get_pretrained(model_id)
        
        # Move model components to correct device
        model.lm = model.lm.to(device)
        model.lm = model.lm.to(torch.float32)  # Convert to float32
    
    # Apply LoRA to the language model
    print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    model.lm = apply_lora(
        model.lm, 
        lora_r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    # Ensure the compression model is also on the same device
    model.compression_model = model.compression_model.to(device)
    
    # Apply CPU offloading if requested
    if cpu_offload:
        print("Enabling CPU offloading for unused parameters")
        try:
            from accelerate import cpu_offload
            for param in model.lm.parameters():
                if not param.requires_grad:
                    cpu_offload(param, device)
        except ImportError:
            print("Warning: accelerate library not found. CPU offloading disabled.")
    
    # Prepare dataset and dataloader
    dataset = AudioDataset(dataset_path, no_label=no_label)
    
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up model for training
    model.lm.train()
    
    # Prepare optimizer - we only need to optimize the LoRA parameters
    optimizer_parameters = [p for p in model.lm.parameters() if p.requires_grad]
    
    if tune_text:
        print("Tuning text conditioner with LoRA")
        optimizer = AdamW(
            model.lm.condition_provider.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
    else:
        print("Tuning language model with LoRA")
        optimizer = AdamW(
            optimizer_parameters,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
    
    # Set up learning rate scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    # Set up loss function and gradient scaler
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Create output directory for model checkpoints
    save_path = "models/lora/"
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize training state
    save_models = False if save_step is None else True
    current_step = 0

    # Main training loop
    for epoch in range(epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            
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
                
            # Prepare conditions
            attributes, _ = model._prepare_tokens_and_attributes(texts, None)
            conditions = attributes
            
            # Apply classifier free guidance if needed
            if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions
                
            # Tokenize and condition the model - ensure tensors are on the right device
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions

            codes = torch.cat(all_codes, dim=0).to(device)

            # Compute model outputs with automatic mixed precision
            if use_scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    lm_output = model.lm.compute_predictions(
                        codes=codes, conditions=[], condition_tensors=condition_tensors
                    )

                    # Extract predictions and targets
                    codes_target = codes[0]
                    logits = lm_output.logits[0]
                    mask = lm_output.mask[0]

                    # Convert to one-hot encoding with gradient retention
                    codes_target = one_hot_encode(codes_target, num_classes=2048)

                    # Move tensors to device and ensure they're differentiable
                    codes_target = codes_target.to(device)
                    logits = logits.to(device)
                    mask = mask.to(device)

                    # Apply mask and compute loss
                    mask = mask.view(-1)
                    masked_logits = logits.view(-1, 2048)[mask]
                    masked_codes = codes_target.view(-1, 2048)[mask]

                    loss = criterion(masked_logits, masked_codes)
            else:
                # Non-autocast version
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                # Extract predictions and targets
                codes_target = codes[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                # Convert to one-hot encoding with gradient retention
                codes_target = one_hot_encode(codes_target, num_classes=2048)

                # Move tensors to device
                codes_target = codes_target.to(device)
                logits = logits.to(device)
                mask = mask.to(device)

                # Apply mask and compute loss
                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes_target.view(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_codes)

            # Update step counter
            current_step += 1 / grad_acc

            # Verify gradients are properly connected
            if not masked_logits.requires_grad:
                print("Warning: masked_logits doesn't require gradients!")
                # Try to find where the gradient chain broke
                for name, param in model.lm.named_parameters():
                    if param.requires_grad:
                        print(f"Parameter requiring grad: {name}")
                
                # Let's make sure the logits require gradients
                if hasattr(lm_output, 'logits_grad_fn'):
                    print("Found logits_grad_fn:", lm_output.logits_grad_fn)
            
            # Backward pass with gradient scaling if enabled
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Compute gradient norm
            total_norm = 0
            for p in optimizer_parameters:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except (AttributeError, TypeError):
                    pass
            total_norm = total_norm ** 0.5

            # Log metrics
            if use_wandb and (not is_distributed or rank == 0):
                run.log(
                    {
                        "loss": loss.item(),
                        "total_norm": total_norm,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # Print progress
            print(
                f"Epoch: {epoch}/{epochs}, Batch: {batch_idx}/{len(train_dataloader)}, "
                f"Loss: {loss.item():.6f}, Grad Norm: {total_norm:.6f}"
            )

            # Skip gradient update if not at accumulation boundary
            if batch_idx % grad_acc != grad_acc - 1:
                continue

            # Gradient clipping
            if use_scaler:
                scaler.unscale_(optimizer)
                
            # Clip gradients to prevent explosions
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, 0.5)

            # Optimizer step
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Update learning rate
            scheduler.step()

            # Save model checkpoint if requested based on steps
            if save_models and int(current_step) % save_step == 0 and (not is_distributed or rank == 0):
                save_checkpoint(model, f"{save_path}/musicgen_lora_{int(current_step)}.pt")
                print(f"Saved checkpoint at step {int(current_step)}")

        # Save checkpoint after each epoch (new code)
        if not is_distributed or rank == 0:
            epoch_save_path = f"{save_path}/musicgen_lora_epoch_{epoch+1}.pt"
            save_checkpoint(model, epoch_save_path)
            print(f"Saved checkpoint after epoch {epoch+1} to {epoch_save_path}")

    # Save final model
    if not is_distributed or rank == 0:
        save_checkpoint(model, f"{save_path}/musicgen_lora_final.pt")
        print("Training complete! Final model saved.")
    
    # Clean up distributed training
    if is_distributed:
        dist.destroy_process_group()


def save_checkpoint(model, path):
    """Save a LoRA checkpoint with only the adapter weights."""
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save LoRA weights only - use model.lm instead of model
    lora_state_dict = {}
    for name, module in model.lm.named_modules():
        if hasattr(module, "lora_A"):
            module_name = name
            lora_state_dict[f"{module_name}.lora_A.weight"] = module.lora_A.weight.data
            lora_state_dict[f"{module_name}.lora_B.weight"] = module.lora_B.weight.data
    
    torch.save({"lora_state_dict": lora_state_dict}, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model, lora_path):
    """Load LoRA weights into a model."""
    if not os.path.exists(lora_path):
        print(f"Error: LoRA weights file not found: {lora_path}")
        return model
        
    state_dict = torch.load(lora_path, map_location="cpu")
    
    if "lora_state_dict" not in state_dict:
        print("Error: Invalid LoRA weights file format")
        return model
        
    lora_state_dict = state_dict["lora_state_dict"]
    
    # Apply weights to modules - use model.lm instead of model
    for name, module in model.lm.named_modules():
        if hasattr(module, "lora_A"):
            module_name = name
            if f"{module_name}.lora_A.weight" in lora_state_dict:
                module.lora_A.weight.data = lora_state_dict[f"{module_name}.lora_A.weight"]
                module.lora_B.weight.data = lora_state_dict[f"{module_name}.lora_B.weight"]
    
    print(f"Loaded LoRA weights from {lora_path}")
    return model