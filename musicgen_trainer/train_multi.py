import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler
from tqdm.auto import tqdm
import wandb
import time
import numpy as np
import gc
import psutil
import random
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import torchaudio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDataset(Dataset):
    def __init__(self, dataset_path, no_label=False):
        self.dataset_path = Path(dataset_path)
        self.audio_files = list(self.dataset_path.glob("**/*.wav"))
        self.no_label = no_label
        
        if not self.no_label:
            # Assuming text files with same name as audio files contain descriptions
            self.text_files = [audio_path.with_suffix('.txt') for audio_path in self.audio_files]
            # Validate that all text files exist
            for txt_file in self.text_files:
                if not txt_file.exists():
                    raise FileNotFoundError(f"Text file not found: {txt_file}")
        
        logger.info(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = audio_read(str(audio_path))
        
        # Process text prompt if available
        if not self.no_label:
            with open(self.text_files[idx], 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""
            
        return {
            "audio": audio,
            "text": text,
            "path": str(audio_path)
        }

def preprocess_audio_for_model(model, audio_list):
    """Robust audio preprocessing for MusicGen that handles different model versions."""
    try:
        # Try different common patterns for audio preprocessing
        if hasattr(model, 'preprocess_audio'):
            # Direct method on model
            return model.preprocess_audio(audio_list)
        elif hasattr(model, 'preprocess'):
            # Generic preprocess method
            return model.preprocess(audio_list)
        elif hasattr(model, 'compression_model') and hasattr(model.compression_model, 'preprocess'):
            # Access through compression model
            return model.compression_model.preprocess(audio_list)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'preprocess_audio'):
            # Access through encoder
            return model.encoder.preprocess_audio(audio_list)
        elif hasattr(model, 'lm') and hasattr(model.lm, 'preprocess_audio'):
            # Access through language model
            return model.lm.preprocess_audio(audio_list)
        else:
            # Fallback: Manually implement basic audio preprocessing
            logger.warning("No preprocessing method found in model, using basic fallback preprocessing")
            
            # Import torch audio if available
            try:
                import torchaudio.transforms as T
                
                # Convert list of audios to tensor ensuring right device
                device = 'cpu'  # Always process on CPU first
                processed = []
                for audio in audio_list:
                    # Convert to float32 to ensure consistency
                    if isinstance(audio, torch.Tensor) and audio.dtype != torch.float32:
                        audio = audio.to(torch.float32)
                    
                    # Normalize if not already done
                    if audio.abs().max() > 1.0:
                        audio = audio / audio.abs().max()
                    
                    # Basic resampling if needed (assumes 44.1kHz for MusicGen)
                    if hasattr(audio, 'sample_rate') and audio.sample_rate != 44100:
                        resampler = T.Resample(audio.sample_rate, 44100)
                        audio = resampler(audio)
                    
                    processed.append(audio)
                
                # Stack into batch if all same shape, otherwise return list
                try:
                    result = torch.stack(processed)
                    logger.info(f"Preprocessed audio batch with shape {result.shape} on {device}")
                    return result
                except:
                    logger.info(f"Returning list of processed audio tensors (couldn't stack)")
                    return processed
                    
            except ImportError:
                # If torchaudio not available, return unprocessed
                logger.error("Could not preprocess audio - torchaudio not available")
                return audio_list
    
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {e}")
        # Return input as fallback with warning
        logger.warning("Using unprocessed audio due to preprocessing error")
        return audio_list

def count_nans(tensor):
    """Count number of NaN values in a tensor"""
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans

def fixnan(tensor: torch.Tensor):
    """Replace NaN values with zeros in a tensor"""
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    return result

def one_hot_encode(tensor, num_classes=2048):
    """Convert integer indices to one-hot encoded vectors"""
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes), device=tensor.device)

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

def collate_fn(batch, model):
    """Collate function that handles preprocessing audio for the model."""
    audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Process audios using the robust preprocessing function
    processed_audios = preprocess_audio_for_model(model, audios)
    
    return {
        "audio": processed_audios,
        "text": texts
    }

def log_memory_usage():
    """Log GPU and CPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**3
    logger.info(f"CPU RAM Usage: {ram_usage:.2f}GB")

def clear_memory():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_trainable_parameters(model):
    """Get trainable parameters from a MusicGen model or other model types.
    Handles different model architectures by trying different approaches.
    """
    params = []
    try:
        # Try direct access if it's a regular nn.Module
        if isinstance(model, nn.Module):
            return [p for p in model.parameters() if p.requires_grad]
        
        # Check for common MusicGen components that might be nn.Modules
        if hasattr(model, 'lm') and isinstance(model.lm, nn.Module):
            logger.info("Getting parameters from model.lm")
            params.extend([p for p in model.lm.parameters() if p.requires_grad])
        
        if hasattr(model, 'compression_model') and isinstance(model.compression_model, nn.Module):
            logger.info("Getting parameters from model.compression_model")
            params.extend([p for p in model.compression_model.parameters() if p.requires_grad])
            
        # Try other common components
        for attr_name in ['encoder', 'decoder', 'model', 'transformer']:
            if hasattr(model, attr_name) and isinstance(getattr(model, attr_name), nn.Module):
                logger.info(f"Getting parameters from model.{attr_name}")
                params.extend([p for p in getattr(model, attr_name).parameters() if p.requires_grad])
        
        # If no parameters found, try to iterate through all attributes
        if not params:
            logger.warning("No standard parameters found, searching through all attributes")
            for attr_name in dir(model):
                attr = getattr(model, attr_name)
                if isinstance(attr, nn.Module) and not attr_name.startswith('_'):
                    logger.info(f"Found module in attribute: {attr_name}")
                    params.extend([p for p in attr.parameters() if p.requires_grad])
        
        if not params:
            raise ValueError("Could not find any trainable parameters in the model")
            
        logger.info(f"Found {len(params)} trainable parameters")
        return params
    
    except Exception as e:
        logger.error(f"Error finding trainable parameters: {e}")
        raise

def set_training_mode(model, training=True):
    """Set training mode for a MusicGen model or its components."""
    mode_str = "training" if training else "evaluation"
    logger.info(f"Setting model to {mode_str} mode")
    
    # If model supports train() directly, use it
    if hasattr(model, 'train') and callable(model.train):
        if training:
            model.train()
        else:
            model.eval()
        return True
    
    # Otherwise, find and set training mode for all nn.Module components
    found_modules = False
    
    # Common component names that might be nn.Modules
    potential_modules = ['lm', 'compression_model', 'encoder', 'decoder', 
                         'transformer', 'model', 'codec', 'audio_encoder']
    
    for name in potential_modules:
        if hasattr(model, name):
            component = getattr(model, name)
            if isinstance(component, nn.Module):
                if training:
                    component.train()
                else:
                    component.eval()
                logger.info(f"Set {mode_str} mode for model.{name}")
                found_modules = True
    
    # If no standard components found, try to traverse all attributes
    if not found_modules:
        for attr_name in dir(model):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(model, attr_name)
                if isinstance(attr, nn.Module):
                    if training:
                        attr.train()
                    else:
                        attr.eval()
                    logger.info(f"Set {mode_str} mode for model.{attr_name}")
                    found_modules = True
            except Exception:
                pass
    
    if not found_modules:
        logger.warning(f"Could not find any nn.Module components to set {mode_str} mode")
        return False
        
    return True

def load_model_on_cpu(model_id):
    """Load MusicGen model on CPU to avoid GPU OOM errors."""
    # Import torch at the beginning of the function
    import torch
    
    logger.info(f"Loading MusicGen model '{model_id}' on CPU first...")
    
    # Store original CUDA device setting
    orig_cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    orig_default_dtype = torch.get_default_dtype()
    
    # Check for PyTorch version compatibility
    has_default_device = hasattr(torch, 'set_default_device')
    
    if has_default_device:
        try:
            orig_device = torch.get_default_device()
            torch.set_default_device('cpu')
        except AttributeError:
            has_default_device = False
    
    # Disable CUDA caching allocator
    orig_allocator_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1'
    
    try:
        # Completely disable CUDA for loading
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Force model loading on CPU
        with torch.device('cpu'):
            # Manually empty cache if GPU is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Disable optimizations that might use GPU
            os.environ['USE_FLASH_ATTENTION'] = '0'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            if isinstance(model_id, str):
                # Import inside function to ensure CUDA settings are applied
                from audiocraft.models import MusicGen
                
                # First check if we're loading a local path or a HF model ID
                if os.path.exists(model_id):
                    logger.info(f"Loading model from local path: {model_id}")
                    try:
                        # Load as a state dict file
                        state_dict = torch.load(model_id, map_location='cpu')
                        
                        # Load base model first then apply weights
                        model = MusicGen.get_pretrained('small', device='cpu')
                        if hasattr(model, 'lm'):
                            model.lm.load_state_dict(state_dict)
                        else:
                            model.load_state_dict(state_dict)
                    except Exception as e:
                        logger.error(f"Failed to load as state dict: {e}, will try as complete model")
                        model = torch.load(model_id, map_location='cpu')
                else:
                    # Load from HF with explicit device specification
                    model = MusicGen.get_pretrained(model_id, device='cpu')
                logger.info(f"Successfully loaded pretrained model '{model_id}' on CPU")
            else:
                model = model_id
                logger.info("Using provided model instance")
        
        return model
    finally:
        # Restore original environment
        os.environ['CUDA_VISIBLE_DEVICES'] = orig_cuda_device
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = orig_allocator_config
        
        # Restore original device if applicable
        if has_default_device:
            try:
                torch.set_default_device(orig_device)
            except:
                pass
        
        # Clean up environment variables
        for var in ['USE_FLASH_ATTENTION', 'TOKENIZERS_PARALLELISM']:
            if os.environ.get(var) == '0' or os.environ.get(var) == 'false':
                os.environ.pop(var, None)

def find_text_components(model):
    """Safe method to find text-related components in a MusicGen model."""
    components = []
    
    # Check for direct text components we know about
    for path, names in [
        (["lm", "text_encoder"], ["text_encoder"]),
        (["encoders", "text"], ["text", "text_encoder"]),
        (["text_model"], ["text_model", "text_encoder"]),
        (["text_encoder"], ["text_encoder"])
    ]:
        # Try to navigate the path
        current = model
        valid_path = True
        for part in path:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                valid_path = False
                break
        
        if valid_path and isinstance(current, nn.Module):
            components.append((f"{'.'.join(path)}", current))
    
    # If no standard paths work, try a generic attribute search for text-related modules
    if not components:
        logger.warning("No standard text encoder found, searching all attributes")
        stack = [([], model)]
        max_depth = 3  # Limit search depth to avoid deep recursion
        
        while stack and len(components) == 0:
            path, current = stack.pop()
            
            # Skip if too deep
            if len(path) >= max_depth:
                continue
                
            # Check all attributes of current object
            for name in dir(current):
                # Skip private attributes
                if name.startswith('_'):
                    continue
                    
                try:
                    attr = getattr(current, name)
                    
                    # Check if this appears to be a text-related component
                    if isinstance(attr, nn.Module) and any(text_name in name.lower() for text_name in ["text", "t5", "llm", "token", "embed"]):
                        full_path = path + [name]
                        components.append((f"{'.'.join(full_path)}", attr))
                        logger.info(f"Found potential text component: {'.'.join(full_path)}")
                    
                    # If it's a nn.Module, queue it for later exploration
                    elif isinstance(attr, nn.Module) and not isinstance(attr, nn.ModuleList) and not isinstance(attr, nn.Sequential):
                        new_path = path + [name]
                        stack.append((new_path, attr))
                except Exception as e:
                    # Skip attributes that cause errors when accessed
                    continue
    
    return components

def setup_multi_gpu(model):
    """Set up for multi-GPU training using DataParallel"""
    if torch.cuda.device_count() > 1:
        logger.info(f"Setting up DataParallel across {torch.cuda.device_count()} GPUs")
        # Wrap model with DataParallel to use multiple GPUs
        model = nn.DataParallel(model)
        return model, True
    else:
        logger.info("Using single GPU for training")
        return model, False

def move_model_to_device(model, device):
    """Aggressively move MusicGen components to specified device"""
    logger.info(f"Aggressively moving MusicGen components to {device}")
    
    # MusicGen doesn't support direct .to() but its components do
    components_moved = 0
    
    # Try to move key components
    for component_name in ['lm', 'compression_model', 'encodec', 'codec', 'transformer', 
                          'condition_provider', 'text_encoder', 'audio_encoder']:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if hasattr(component, 'to'):
                try:
                    component_device = next(component.parameters()).device
                    logger.info(f"Component {component_name} was on {component_device}")
                    getattr(model, component_name).to(device)
                    components_moved += 1
                    new_device = next(component.parameters()).device
                    logger.info(f"Moved {component_name} to {new_device}")
                except Exception as e:
                    logger.error(f"Failed to move {component_name}: {e}")
    
    # Handle nested components 
    if hasattr(model, 'lm'):
        lm = model.lm
        lm_components_moved = 0
        for lm_comp in ['condition_provider', 'transformer', 'text_encoder', 'heads']:
            if hasattr(lm, lm_comp):
                comp = getattr(lm, lm_comp)
                if hasattr(comp, 'to'):
                    try:
                        comp.to(device)
                        lm_components_moved += 1
                        logger.info(f"Moved lm.{lm_comp} to {device}")
                    except Exception as e:
                        logger.error(f"Failed to move lm.{lm_comp}: {e}")
        
        logger.info(f"Moved {lm_components_moved} LM components to {device}")
    
    logger.info(f"Successfully moved {components_moved} components to {device}")
    return components_moved > 0

def train_multi(
    dataset_path,
    model_id="small",
    lr=1e-5,
    epochs=100,
    use_wandb=0,
    save_step=None,
    no_label=0,
    tune_text=0,
    weight_decay=1e-5,
    grad_acc=4,     # Increased gradient accumulation
    warmup_steps=16,
    batch_size=2,   # Smaller batch size
    use_cfg=0,
    devices=None,
    gradient_checkpointing=1,
    memory_efficient_attention=1,
    mixed_precision='bf16',
    cpu_offload=1,
    pin_memory=True,
    offload_dir="./offload",
    use_8bit=False,
    max_memory=None,
    offload_optimizer=True,
    offload_param_after_fwd=True,
    memory_monitor_interval=10,
    checkpoint_cpu=True,      # Save checkpoints to CPU first
    force_cpu_loading=True,   # Force loading model on CPU first
    use_scaler=True,          # Enable gradient scaling for better numerical stability
    use_direct_loss=False,    # Use direct loss from model.loss or manual calculation
    multi_gpu=True,           # Enable multi-GPU training if available
    restore_devices=None,     # New parameter to restore GPU visibility
    # Add the missing parameters with appropriate defaults
    force_gpu=False,
    allow_tf32=False,
):
    global logger
    # Set up logging
    # Create offload directory if needed
    if cpu_offload and offload_dir:
        os.makedirs(offload_dir, exist_ok=True)
    
    # Set environment variable to avoid memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # REMOVED: Multi-GPU setup code - moved to after model loading
    
    # Set max memory mapping - handle all GPUs, not just GPU 0
    if max_memory is None and torch.cuda.is_available():
        max_memory = {"cpu": "64GiB"}  # Default CPU memory
        
        # Get devices to check - either from devices parameter or all available
        gpus_to_check = devices if devices else list(range(torch.cuda.device_count()))
        logger.info(f"Checking memory for GPUs: {gpus_to_check}")
        
        for gpu_id in gpus_to_check:
            # Calculate available memory for each GPU
            available_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            
            # Get current reserved memory (a better estimate than allocated memory)
            torch.cuda.set_device(gpu_id)  # Set device context for memory check
            free_mem = torch.cuda.memory_reserved(gpu_id) / 1024**3
            free_mem = available_mem - free_mem  # Calculate actual free memory
            
            logger.info(f"GPU {gpu_id}: {available_mem:.1f}GB total, {free_mem:.1f}GB free")
            
            # Use a conservative value based on currently available memory
            safe_mem = min(available_mem * 0.7, free_mem * 0.9) if free_mem > 0 else available_mem * 0.7
            max_memory[gpu_id] = f"{int(safe_mem)}GiB"
        
        logger.info(f"Auto-configured memory map: {max_memory}")
    
    # Initialize accelerator with compatible parameters
    logger.info("Initializing accelerator")
    logger = get_logger(__name__)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_acc,
        mixed_precision=mixed_precision,
        device_placement=True,
        cpu=force_cpu_loading,  # Start with CPU if needed
    )
    logger.info("Accelerator initialized")
    
    # Log CPU offloading status
    if cpu_offload:
        logger.info("CPU offloading enabled via memory management techniques")
    
    # Clear any existing cached memory before loading model
    clear_memory()
    log_memory_usage()
    
    # Explicitly disable GPU access during loading phase if force_cpu_loading
    original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if force_cpu_loading:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    try:
        # Load model on CPU first if requested
        if force_cpu_loading:
            model = load_model_on_cpu(model_id)
            # Restore GPU access after loading
            if original_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
        else:
            # Load model using standard approach
            if isinstance(model_id, str):
                try:
                    logger.info(f"Loading MusicGen model: {model_id}")
                    model = MusicGen.get_pretrained(model_id, device='auto')
                    logger.info(f"Loaded pretrained model: {model_id}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU OOM during model loading, falling back to CPU loading")
                    model = load_model_on_cpu(model_id)
            else:
                model = model_id
                logger.info("Using provided model instance")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Falling back to CPU loading after error")
        if original_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
        model = load_model_on_cpu(model_id)
    
    # ADDED: Set up multi-GPU environment after model is loaded
    is_multi_gpu = False
    if multi_gpu and devices and len(devices) > 1:
        model, is_multi_gpu = setup_multi_gpu(model)
        logger.info(f"Multi-GPU setup complete: {is_multi_gpu}")
    
    # Set a fixed random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Force PyTorch to use CUDA if available, but don't change defaults
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for improved GPU performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("CUDA is available and will be used for training")
        
        # Set device without changing global defaults
        device_id = 0 if not devices else devices[0]
        torch.cuda.set_device(device_id)
        logger.info(f"Set current CUDA device to {device_id}")
    else:
        logger.error("CUDA is not available! Training may be very slow.")
        
    # CRITICAL: Restore GPU visibility BEFORE creating the accelerator
    if restore_devices:
        logger.info(f"Restoring GPU visibility to: {restore_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = restore_devices
    elif original_visible_devices:
        logger.info(f"Restoring original GPU visibility: {original_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
    
    # Verify CUDA is available after restoration
    if torch.cuda.is_available():
        logger.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("CUDA is still not available after restoring devices! Check your environment.")
        
    # Re-initialize accelerator WITH gpu placement and WITHOUT cpu flag
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_acc,
        mixed_precision=mixed_precision,
        device_placement=True,
        cpu=False,  # Critical: Do not force CPU for training
    )
    
    log_memory_usage()
    
    # Apply memory optimizations
    if gradient_checkpointing:
        try:
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    # Handle memory efficient attention based on actual model structure
    if memory_efficient_attention:
        try:
            # Check if the model has this method or if we need to access components
            if hasattr(model, "use_efficient_attention"):
                model.use_efficient_attention()
                logger.info("Memory efficient attention enabled")
            elif hasattr(model, "lm") and hasattr(model.lm, "transformer"):
                # Try to set attention mechanism on transformer model
                logger.info("Attempting to enable efficient attention on transformer component")
                # This is just a placeholder - actual implementation depends on model structure
                # model.lm.transformer.use_flash_attention = True
            else:
                logger.warning("Could not identify components to enable memory efficient attention")
        except Exception as e:
            logger.warning(f"Failed to enable memory efficient attention: {e}")
    
    # Freeze text encoder if not tuning text - adapted for actual model structure
    if not tune_text:
        try:
            # Use the safer method to find text components
            text_components = find_text_components(model)
            
            if text_components:
                for path, component in text_components:
                    logger.info(f"Freezing parameters for {path}")
                    for param in component.parameters():
                        param.requires_grad = False
            else:
                logger.warning("Could not identify text encoder components. All parameters will be tuned.")
        except Exception as e:
            logger.warning(f"Error while trying to freeze text encoder: {e}")
            logger.warning("Proceeding with all parameters unfrozen")

    # Properly configure the DataLoader's generator
    # FIXED: Always use CPU generator for DataLoader to avoid device type mismatch
    generator = torch.Generator()  # CPU generator is always safe
    generator.manual_seed(seed)
    logger.info("Using CPU random number generator for DataLoader")
    
    # Prepare dataset
    dataset = MusicDataset(dataset_path, no_label=bool(no_label))
    
    # Use a smaller number of workers to reduce memory pressure
    num_workers = 1 if cpu_offload else 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        collate_fn=lambda b: collate_fn(b, model),
        persistent_workers=False,  # Disable persistent workers to save memory
        generator=generator  # Use CPU generator
    )
    
    # Identify trainable parameters
    trainable_params = get_trainable_parameters(model)
    
    # Setup optimizer with memory considerations
    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW")
            optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
    else:
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
    
    # Setup scheduler
    num_training_steps = len(dataloader) * epochs // grad_acc
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Setup wandb
    if use_wandb:
        wandb.init(
            project="musicgen-finetune",
            config={
                "model_id": model_id if isinstance(model_id, str) else "custom_model",
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_acc": grad_acc,
                "tune_text": tune_text,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "cpu_offload": cpu_offload,
                "mixed_precision": mixed_precision,
            }
        )
    
    # Move model to GPU explicitly before passing to accelerator (belt and suspenders approach)
    if torch.cuda.is_available():
        try:
            device_id = 0 if not devices else devices[0]
            logger.info(f"Moving model to CUDA device {device_id} before accelerator preparation")
            model = model.to(f"cuda:{device_id}")
        except Exception as e:
            logger.error(f"Error moving model to GPU: {e}")
    
    # Fix model placement before accelerator preparation
    if torch.cuda.is_available():
        logger.info("Ensuring model components are on CUDA before accelerator preparation")
        device_id = 0 if not devices else devices[0]
        cuda_device = torch.device(f"cuda:{device_id}")
        
        # Try to force model components to GPU directly
        try:
            # Handle MusicGen's special structure
            if hasattr(model, 'lm'):
                model.lm.to(cuda_device)
                logger.info(f"Moved model.lm to {cuda_device}")
            if hasattr(model, 'compression_model'):
                model.compression_model.to(cuda_device)
                logger.info(f"Moved model.compression_model to {cuda_device}")
        except Exception as e:
            logger.error(f"Error pre-moving model components: {e}")
    
    # IMPORTANT: Remove the model.to() call since MusicGen doesn't support it
    
    # Ensure we're using correct device settings in accelerator
    logger.info("Preparing model with accelerator (moving to device if possible)")
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU for training")
    
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # IMPORTANT: Check model components directly
    device = accelerator.device
    logger.info(f"Accelerator device: {device}")
    
    component_devices = []
    if hasattr(model, 'lm'):
        try:
            lm_device = next(model.lm.parameters()).device
            component_devices.append(('lm', lm_device))
        except:
            pass
    if hasattr(model, 'compression_model'):
        try:
            cm_device = next(model.compression_model.parameters()).device
            component_devices.append(('compression_model', cm_device))
        except:
            pass
            
    logger.info(f"Model component devices: {component_devices}")
    
    # If components are still on CPU, try one more direct placement
    if torch.cuda.is_available() and any(device.type == 'cpu' for _, device in component_devices):
        logger.warning("Some model components still on CPU, attempting final GPU transfer")
        move_model_to_device(model, torch.device('cuda'))
        
        # Re-check after move attempt
        component_devices = []
        if hasattr(model, 'lm'):
            lm_device = next(model.lm.parameters()).device
            component_devices.append(('lm', lm_device))
        if hasattr(model, 'compression_model'):
            cm_device = next(model.compression_model.parameters()).device
            component_devices.append(('compression_model', cm_device))
        logger.info(f"After final move attempt, component devices: {component_devices}")
    
    # Only create gradient scaler if CUDA is available and we're using mixed precision
    can_use_scaler = device.type == 'cuda' and use_scaler and mixed_precision and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if can_use_scaler else None
    logger.info(f"Using gradient scaling: {scaler is not None}")
    
    # Add extra offloading for optimizer states if needed
    if cpu_offload and offload_optimizer:
        try:
            logger.info("Moving optimizer states to CPU to save GPU memory")
            # Manual implementation for moving optimizer states to CPU
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param in optimizer.state:
                        for key, value in optimizer.state[param].items():
                            if torch.is_tensor(value):
                                optimizer.state[param][key] = value.cpu()
            logger.info("Successfully offloaded optimizer states to CPU")
        except Exception as e:
            logger.warning(f"Failed to offload optimizer states to CPU: {e}")
    
    # Create criterion for loss calculation - ensure it's on the right device
    criterion = nn.CrossEntropyLoss().to(accelerator.device)
    
    # Training loop
    logger.info("Starting training loop")
    global_step = 0
    
    output_dir = Path("./checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Use the helper function instead of calling model.train() directly
        set_training_mode(model, training=True)
        epoch_start = time.time()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        
        for step, batch in enumerate(progress_bar):
            # Monitor memory periodically
            if step % memory_monitor_interval == 0:
                log_memory_usage()
            
            with accelerator.accumulate(model):
                # Get audio and text inputs
                audio_tensors = batch["audio"]
                text_inputs = batch["text"]
                
                # Important: verify tensor device before processing
                try:
                    # Always explicitly move tensors to the right device before processing
                    if isinstance(audio_tensors, torch.Tensor):
                        if audio_tensors.device != accelerator.device:
                            logger.info(f"Moving audio tensor from {audio_tensors.device} to {accelerator.device}")
                            audio_tensors = audio_tensors.to(accelerator.device)
                    
                    # Process with appropriate context manager based on device and settings
                    # ...existing code...
                    
                    # Inside the autocast context:
                    with ctx_manager:
                        # Explicitly check and ensure compression model is on GPU
                        try:
                            if hasattr(model.compression_model, 'to'):
                                if not check_component_device(model.compression_model, accelerator.device):
                                    logger.warning(f"Moving compression_model to {accelerator.device} before encode")
                                    model.compression_model.to(accelerator.device)
                        except Exception as e:
                            logger.error(f"Failed to verify compression_model device: {e}")
                            
                        # Now encode audio with explicit device checking
                        if hasattr(model, 'compression_model') and hasattr(model.compression_model, 'encode'):
                            # Verify input and weight devices match before encode
                            device_info = f"Audio tensor: {audio_tensors.device}"
                            try:
                                cm_param = next(model.compression_model.parameters())
                                device_info += f", Compression model: {cm_param.device}"
                            except:
                                pass
                            logger.info(f"Device check before encode: {device_info}")
                            
                            # Try to avoid device mismatch error by ensuring same device
                            if isinstance(audio_tensors, torch.Tensor) and audio_tensors.device.type != 'cuda' and accelerator.device.type == 'cuda':
                                logger.info(f"Moving audio to cuda before encode")
                                audio_tensors = audio_tensors.to(accelerator.device)
                            
                            gen_audio = model.compression_model.encode(audio_tensors)
                            codes, scale = gen_audio
                            
                            # For CFG if needed (make a copy, don't detach yet)
                            if use_cfg:
                                codes = torch.cat([codes, codes], dim=0)
                        else:
                            logger.error("Cannot find compression_model.encode method")
                            raise AttributeError("Model lacks proper compression_model.encode method")
                        
                        # Process text conditioning - standard pipeline
                        attributes, _ = model._prepare_tokens_and_attributes(text_inputs, None)
                        conditions = attributes
                        
                        # Apply classifier-free guidance if requested
                        if use_cfg:
                            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                            conditions = conditions + null_conditions
                        
                        # Tokenize and process conditions
                        tokenized = model.lm.condition_provider.tokenize(conditions)
                        cfg_conditions = model.lm.condition_provider(tokenized)
                        
                        # IMPORTANT: We only want to train the LM, not the compression model
                        # Detach codes here to prevent gradients from flowing back to compression model
                        # But we need to maintain the computation graph for the LM
                        if codes.requires_grad:
                            codes_for_lm = codes.detach()
                        else:
                            codes_for_lm = codes  # Already detached
                        
                        # Compute predictions from language model - this needs to be differentiable
                        lm_output = model.lm.compute_predictions(
                            codes=codes_for_lm,
                            conditions=[],
                            condition_tensors=cfg_conditions
                        )
                        
                        # Extract batch elements - these should have grad_fn if forward pass is correct
                        batch_codes = codes_for_lm[0]
                        logits = lm_output.logits[0]
                        mask = lm_output.mask[0]
                        
                        # Convert codes to targets that criterion can use (CrossEntropyLoss expects class indices not one-hot)
                        # Original approach with one-hot encoding:
                        # one_hot_codes = one_hot_encode(batch_codes, num_classes=2048)
                        # one_hot_codes = one_hot_codes.to(accelerator.device)
                        # masked_codes = one_hot_codes.view(-1, 2048)[mask]
                        
                        # Better approach: use class indices directly (more efficient)
                        batch_codes_flat = batch_codes.view(-1)
                        masked_codes_idx = batch_codes_flat[mask]
                        
                        # Prepare logits
                        logits = logits.to(accelerator.device)
                        mask = mask.to(accelerator.device)
                        masked_logits = logits.view(-1, 2048)[mask]
                        
                        # Calculate loss with indices (CrossEntropyLoss expects logits and target class indices)
                        loss = criterion(masked_logits, masked_codes_idx)
                        
                        # Verify loss has grad_fn
                        if not hasattr(loss, 'grad_fn'):
                            logger.error("Loss doesn't have grad_fn, cannot backpropagate")
                            logger.error(f"Loss: {loss}, requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else None}")
                            continue
                except Exception as e:
                    logger.error(f"Error during forward pass: {e}")
                    # Print device debugging info
                    log_memory_usage()
                    if isinstance(audio_tensors, torch.Tensor):
                        logger.error(f"  audio_tensors device: {audio_tensors.device}, shape: {audio_tensors.shape}")
                    if 'model' in locals() and hasattr(model, 'compression_model'):
                        try:
                            logger.error(f"  model.compression_model device: {next(model.compression_model.parameters()).device}")
                        except:
                            pass
                    continue
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning("NaN loss detected, skipping batch")
                    continue
                
                # Backward pass with appropriate handling based on device/settings
                if can_use_scaler:
                    scaler.scale(loss).backward()
                    # Unscale before gradient clipping
                    if trainable_params:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    # Update with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass with accelerator
                    accelerator.backward(loss)
                    # Clip gradients if needed
                    if trainable_params:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    # Standard optimizer step
                    optimizer.step()
                
                # Update learning rate
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update progress metrics
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                if use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    })
                
                global_step += 1
                
                # Save model at specified steps
                if save_step and global_step % save_step == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    
                    if checkpoint_cpu:
                        logger.info(f"Moving model to CPU for saving checkpoint...")
                        with accelerator.main_process_first():
                            unwrapped_model = accelerator.unwrap_model(model)
                            cpu_state_dict = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}
                            accelerator.save(cpu_state_dict, f"{save_path}/model.pt")
                    else:
                        accelerator.save_state(save_path)
                        
                    logger.info(f"Saved checkpoint at step {global_step} to {save_path}")
                    
                    # Clear memory after saving
                    clear_memory()
        
        # Save model after each epoch
        epoch_save_path = output_dir / f"epoch-{epoch+1}"
        accelerator.save_state(epoch_save_path)
        logger.info(f"Saved checkpoint for epoch {epoch+1} to {epoch_save_path}")
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
        
        if use_wandb:
            wandb.log({"epoch": epoch, "avg_loss": avg_loss, "epoch_time": epoch_time})
        
        # Clear memory between epochs
        clear_memory()
        log_memory_usage()
    
    # When saving the final model, might want to set eval mode
    set_training_mode(model, training=False)
    
    # Save final model
    final_save_path = output_dir / "final-model"
    accelerator.save_state(final_save_path)
    logger.info(f"Training complete. Final model saved to {final_save_path}")
    
    # Copy the model to models directory for compatibility with run_all.py
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs("models", exist_ok=True)
    accelerator.save(unwrapped_model.state_dict(), "models/lm_final.pt")
    logger.info("Model saved to models/lm_final.pt")
    
    if use_wandb:
        wandb.finish()
    
    # Final cleanup
    clear_memory()
    
    return model

def check_component_device(component, target_device):
    """Check if a model component is on the target device"""
    try:
        param = next(component.parameters())
        return param.device == target_device
    except:
        return False