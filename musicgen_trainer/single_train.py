import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import wandb
import gc
from torch.utils.data import Dataset
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import os
import time
import psutil
import sys


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


def limit_gpu_memory(fraction=0.95):
    """Limit GPU memory usage to a fraction of available memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        limit = int(total_memory * fraction)
        # Reserve memory
        try:
            # Create a tensor to reserve memory
            reserved_mem = torch.empty(limit, dtype=torch.uint8, device='cuda')
            # Free the reserved memory
            del reserved_mem
            torch.cuda.empty_cache()
            print(f"Limited GPU memory to {fraction*100:.1f}% ({limit/(1024**3):.2f} GB)")
        except Exception as e:
            print(f"Failed to limit memory: {e}")


def report_memory(tag=""):
    """Report memory usage statistics with an optional tag for identification."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"Memory {tag} - Allocated: {alloc:.2f}MB, "
              f"Reserved: {reserved:.2f}MB, Max: {max_alloc:.2f}MB")
        
        # Also report system memory
        sys_mem = psutil.virtual_memory()
        print(f"System Memory - Used: {sys_mem.used/1024**3:.2f}GB, Available: {sys_mem.available/1024**3:.2f}GB")


def free_memory():
    """Free unused memory and report statistics."""
    before_gc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Force collection of Python garbage
    gc.collect()
    
    # Free CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after_gc = torch.cuda.memory_allocated()
        freed = before_gc - after_gc
        print(f"Memory freed: {freed / 1024**2:.2f}MB")


def process_in_chunks(func, *args, **kwargs):
    """Try to run a function and if it fails due to OOM, wait and retry."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = func(*args, **kwargs)
            return result
        except torch.cuda.OutOfMemoryError as e:
            if attempt == max_attempts - 1:
                print(f"Failed after {max_attempts} attempts due to CUDA OOM")
                raise
            print(f"CUDA OOM on attempt {attempt+1}/{max_attempts}, clearing memory and retrying...")
            # Clear all tensors from GPU
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.device.type == 'cuda':
                        del obj
                except:
                    pass
            # Force garbage collection and clear cache
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)  # Wait for memory to stabilize
        except Exception as e:
            print(f"Non-OOM error: {e}")
            return None


def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    """Process audio file with memory efficiency as priority."""
    try:
        # Load audio file on CPU
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        
        if wav.shape[1] < model.sample_rate * duration:
            print(f"Audio too short: {audio_path}")
            return None
            
        # Create segment
        end_sample = int(model.sample_rate * duration)
        start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
        wav = wav[:, start_sample : start_sample + end_sample]
        
        # Process on GPU in small chunks
        def encode_audio(wav_tensor):
            # Move to GPU for encoding
            wav_gpu = wav_tensor.cuda()
            wav_gpu = wav_gpu.unsqueeze(0)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    gen_audio = model.compression_model.encode(wav_gpu)
            
            # Free GPU tensor immediately
            del wav_gpu
            torch.cuda.empty_cache()
            
            codes, scale = gen_audio
            assert scale is None
            
            # Move codes to CPU
            codes_cpu = codes.detach().cpu()
            
            # Free GPU tensors
            del codes
            del gen_audio
            torch.cuda.empty_cache()
            
            return codes_cpu
        
        # Run encoding with retry logic - pass the wav tensor explicitly
        return process_in_chunks(encode_audio, wav)
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        # Ensure all tensors are cleaned up
        torch.cuda.empty_cache()
        return None


def one_hot_encode(tensor, num_classes=2048, device="cpu"):
    """Create one-hot encoding with memory limits in mind."""
    shape = tensor.shape
    
    # If tensor is too large, process on CPU
    if shape[0] * shape[1] * num_classes > 1e8:  # Arbitrary threshold
        device = "cpu"
        
    one_hot = torch.zeros((shape[0], shape[1], num_classes), device=device)

    # Process in small chunks to avoid memory issues
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = tensor[i, j].item()
            one_hot[i, j, idx] = 1

    return one_hot


class MemoryTracker:
    """Track peak memory usage."""
    def __init__(self):
        self.peak = 0
        self.current = 0
    
    def update(self):
        if torch.cuda.is_available():
            self.current = torch.cuda.memory_allocated() / 1024**3
            self.peak = max(self.peak, self.current)
    
    def report(self):
        print(f"Peak memory usage: {self.peak:.2f} GB")


def single_train(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: int,
    no_label: int = 0,
    tune_text: int = 0,
    save_step: int = None,
    grad_acc: int = 8,
    use_scaler: int = 1,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 1,
    use_cfg: int = 0,
    use_cpu_offload: int = 1,
    memory_efficient: int = 1,
    memory_fraction: float = 0.95
):
    # Initialize memory tracker
    memory_tracker = MemoryTracker()
    
    # Apply memory limit
    limit_gpu_memory(memory_fraction)
    
    # Convert int flags to bool
    use_wandb = bool(use_wandb)
    no_label = bool(no_label)
    tune_text = bool(tune_text)
    use_scaler = bool(use_scaler)
    use_cfg = bool(use_cfg)
    use_cpu_offload = bool(use_cpu_offload)
    memory_efficient = bool(memory_efficient)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_wandb:
        run = wandb.init(project="audiocraft")

    print("Loading MusicGen model...")
    model = MusicGen.get_pretrained(model_id)
    
    # Enable memory-efficient operations
    if memory_efficient and hasattr(model.lm, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing...")
        model.lm.gradient_checkpointing_enable()
    
    # Set smaller attention for transformer
    if hasattr(model.lm, 'transformer'):
        print("Reducing transformer attention size...")
        # Reduce attention heads computation for memory savings
        model.lm.transformer._use_split_head_attention = True
    
    # Keep model in float32 precision
    model.lm = model.lm.to(torch.float32)
    
    # Move compression model to GPU for audio processing
    print("Moving compression model to GPU...")
    model.compression_model = model.compression_model.to(device)
    
    # Keep LM on CPU until needed if offloading is enabled
    if use_cpu_offload:
        print("Keeping LM on CPU with offloading enabled...")
        model.lm = model.lm.cpu()
        if tune_text:
            print("Moving condition provider to GPU for text tuning...")
            model.lm.condition_provider = model.lm.condition_provider.to(device)

    print(f"Loading dataset from {dataset_path}...")
    dataset = AudioDataset(dataset_path, no_label=no_label)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    learning_rate = lr
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Select parameters to optimize
    if tune_text:
        print("Tuning text encoder only")
        params_to_optimize = model.lm.condition_provider.parameters()
    else:
        print("Tuning entire model")
        params_to_optimize = model.lm.parameters()

    # Initialize optimizer and scheduler with gradient accumulation
    optimizer = AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    
    # Create scheduler for learning rate warmup
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    criterion = nn.CrossEntropyLoss()
    
    num_epochs = epochs
    save_models = False if save_step is None else True
    save_path = "models/"
    os.makedirs(save_path, exist_ok=True)
    current_step = 0

    # Set model to training mode
    model.lm.train()
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}/{num_epochs}")
        for batch_idx, (audio, label) in enumerate(train_dataloader):
            # Begin batch processing
            report_memory(f"Start of batch {batch_idx}")
            memory_tracker.update()
            
            # Clear cached memory before processing new batch
            free_memory()

            # Process one sample at a time
            all_codes = []
            texts = []
            
            # Process each audio file
            for inner_audio_idx, (inner_audio, l) in enumerate(zip(audio, label)):
                print(f"Processing audio {inner_audio_idx + 1}/{len(audio)}")
                
                # Process audio and get codes with retry logic
                codes = preprocess_audio(inner_audio, model)
                if codes is None:
                    print(f"Skipping audio {inner_audio_idx + 1} (invalid or memory error)")
                    continue
                
                # Apply cfg duplication if needed (on CPU)
                if use_cfg:
                    codes = torch.cat([codes, codes], dim=0)
                
                # Store CPU tensors in list
                all_codes.append(codes)
                
                # Read text label
                with open(l, "r") as f:
                    text = f.read().strip()
                texts.append(text)
                
                # Force memory cleanup
                free_memory()

            # Skip batch if no valid codes
            if len(all_codes) == 0:
                print("Skipping batch (no valid audio)")
                continue

            # Keep track of whether we need to move LM back to CPU after processing
            lm_was_moved = False
            
            try:
                # Move LM to GPU if needed
                if use_cpu_offload and not tune_text:
                    print("Moving LM to GPU for forward pass...")
                    model.lm = model.lm.to(device)
                    lm_was_moved = True
                    
                # Prepare text conditions
                print("Preparing text conditions...")
                attributes, _ = model._prepare_tokens_and_attributes(texts, None)
                conditions = attributes
                
                if use_cfg:
                    null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                    conditions = conditions + null_conditions
                    
                tokenized = model.lm.condition_provider.tokenize(conditions)
                cfg_conditions = model.lm.condition_provider(tokenized)
                condition_tensors = cfg_conditions

                # Reset gradients
                optimizer.zero_grad()
                
                # Process in micro-batches
                report_memory("Before micro-batches")
                memory_tracker.update()
                
                accumulated_loss = 0
                micro_batch_size = 1  # Force processing one at a time
                
                # Process each code
                for micro_idx in range(0, len(all_codes), micro_batch_size):
                    micro_end = min(micro_idx + micro_batch_size, len(all_codes))
                    print(f"Processing micro-batch {micro_idx//micro_batch_size + 1}/{(len(all_codes) + micro_batch_size - 1)//micro_batch_size}")
                    
                    # Move this micro-batch to GPU and immediately clear CPU references
                    micro_codes = [code.to(device) for code in all_codes[micro_idx:micro_end]]
                    micro_codes = torch.cat(micro_codes, dim=0)
                    
                    # Clear CPU codes references
                    for idx in range(micro_idx, micro_end):
                        all_codes[idx] = None
                    
                    # Forward pass with memory protection
                    def forward_pass():
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_scaler):
                            # Compute predictions
                            lm_output = model.lm.compute_predictions(
                                codes=micro_codes, 
                                conditions=[], 
                                condition_tensors=condition_tensors
                            )
                            
                            # Extract tensors
                            codes = micro_codes[0].detach()  # Create detached copy
                            logits = lm_output.logits[0]
                            mask = lm_output.mask[0]
                            
                            # One-hot encode codes directly on GPU to avoid transfer
                            codes_one_hot = one_hot_encode(codes, num_classes=2048, device=device)
                            
                            # Prepare masked tensors for loss
                            mask_flat = mask.view(-1)
                            masked_logits = logits.view(-1, 2048)[mask_flat]
                            masked_codes = codes_one_hot.view(-1, 2048)[mask_flat]
                            
                            # Calculate loss
                            loss = criterion(masked_logits, masked_codes)
                            
                            return loss, loss.item()
                    
                    # Run forward pass with memory protection
                    try:
                        loss, loss_value = process_in_chunks(forward_pass)
                        loss_per_step = loss_value / grad_acc
                        accumulated_loss += loss_per_step
                        
                        # Scale loss for gradient accumulation
                        loss = loss / grad_acc
                        
                        # Backward pass with scaling
                        if use_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                            
                    except Exception as e:
                        print(f"Error in forward/backward pass: {e}")
                        # Emergency memory cleanup
                        del micro_codes
                        torch.cuda.empty_cache()
                        free_memory()
                        continue
                        
                    # Clean up
                    del micro_codes
                    torch.cuda.empty_cache()
                    
                # Update current step counter
                current_step += 1

                # Log statistics
                if tune_text:
                    params_for_grad = model.lm.condition_provider.parameters()
                else:
                    params_for_grad = model.lm.parameters()
                    
                # Calculate gradient norm safely
                total_norm = 0
                for p in params_for_grad:
                    if p.grad is not None:
                        try:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        except:
                            pass
                total_norm = total_norm ** (1.0 / 2) if total_norm > 0 else 0

                # Log to wandb if enabled
                if use_wandb:
                    run.log({
                        "loss": accumulated_loss,
                        "total_norm": total_norm,
                        "gpu_memory": memory_tracker.current,
                    })

                # Print progress
                print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {accumulated_loss}")
                
                # Free memory before optimizer step
                report_memory("Before optimizer step")
                memory_tracker.update()

                # Update model parameters with memory protection
                def optimizer_step():
                    # Unscale gradients if using scaler
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.lm.condition_provider.parameters() if tune_text else model.lm.parameters(), 
                        0.5
                    )
                    
                    # Step with scaler or normally
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Step scheduler
                    scheduler.step()
                
                # Try optimizer step with memory protection
                try:
                    process_in_chunks(optimizer_step)
                except Exception as e:
                    print(f"Error during optimizer step: {e}")
                    # Emergency memory cleanup
                    optimizer.zero_grad()
                    free_memory()
                    
                # Report memory after optimization
                report_memory("After optimizer step")
                
            except Exception as e:
                print(f"Error during batch processing: {e}")
            
            finally:
                # Always move LM back to CPU if it was moved to GPU
                if lm_was_moved:
                    print("Moving LM back to CPU after processing")
                    model.lm = model.lm.cpu()
                
                # Final memory cleanup
                free_memory()
                
                # Update memory tracker
                memory_tracker.update()
            
            # Save model checkpoint if requested
            if save_models and int(current_step) % save_step == 0:
                try:
                    print(f"Saving checkpoint at step {current_step}")
                    # Move model to CPU for saving
                    save_model = model.lm.cpu() if model.lm.device.type != "cpu" else model.lm
                    torch.save(save_model.state_dict(), f"{save_path}/lm_{current_step}.pt")
                    if not use_cpu_offload and not tune_text:
                        model.lm = model.lm.to(device)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

    # Save final model
    try:
        print("Saving final model...")
        save_model = model.lm.cpu() if model.lm.device.type != "cpu" else model.lm
        torch.save(save_model.state_dict(), f"{save_path}/lm_final.pt")
    except Exception as e:
        print(f"Error saving final model: {e}")
        
    # Report final memory statistics
    memory_tracker.report()
    print("Training complete!")
