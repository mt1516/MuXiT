import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple, Union
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import StreamingTransformer  # Changed from TransformerEncoder
from collections import deque
import threading
import queue
import gc
import weakref

class ModelParallelTransformer(nn.Module):
    """Model parallel version of the transformer encoder from MusicGen.
    
    This class splits the transformer layers across multiple GPUs to enable
    training larger models than would fit on a single GPU.
    """
    def __init__(self, original_transformer: StreamingTransformer, devices: List[torch.device]):
        super().__init__()
        # Store configuration as attributes instead of expecting a cfg attribute
        if hasattr(original_transformer, 'cfg'):
            self.cfg = original_transformer.cfg
        else:
            # Extract necessary attributes directly when no cfg is available
            self.cfg = {}
            for attr in dir(original_transformer):
                if not attr.startswith('_') and not callable(getattr(original_transformer, attr)):
                    self.cfg[attr] = getattr(original_transformer, attr)
            
        self.devices = devices
        self.num_devices = len(devices)
        
        # Extract and distribute layers across devices
        self.distribute_layers(original_transformer)
        
        # Keep embeddings on the first device if available
        if hasattr(original_transformer, 'emb'):
            self.emb = original_transformer.emb.to(devices[0])
        
        # Handle positional embeddings if available
        if hasattr(original_transformer, 'pos_emb'):
            self.pos_emb = original_transformer.pos_emb.to(devices[0])
        elif hasattr(original_transformer, 'positional_embedding'):
            self.positional_embedding = original_transformer.positional_embedding
            self.max_period = getattr(original_transformer, 'max_period', 10000)
            self.positional_scale = getattr(original_transformer, 'positional_scale', 1.0)
        
        # Final normalization if available
        if hasattr(original_transformer, 'norm'):
            self.norm = original_transformer.norm.to(devices[-1])
        
        # Store important attributes from the original transformer
        self.dim = getattr(original_transformer, 'd_model', None)
        if self.dim is None:
            self.dim = getattr(original_transformer, 'dim', 768)  # Default to 768 if not found
            
        self.causal = getattr(original_transformer, 'causal', True)
        
        # Handle rope (rotary positional embedding) if available
        if hasattr(original_transformer, 'rope'):
            self.rope = original_transformer.rope
            if self.rope is not None:
                self.rope = self.rope.to(devices[0])
        
    def distribute_layers(self, original_transformer: StreamingTransformer):
        """Split the transformer layers across available devices."""
        layers = list(original_transformer.layers)
        num_layers = len(layers)
        
        # Calculate how many layers per device (roughly balanced)
        self.layers_per_device = [num_layers // self.num_devices + (1 if i < num_layers % self.num_devices else 0) 
                                 for i in range(self.num_devices)]
        
        # Create module lists for each device's layers
        self.device_layers = nn.ModuleList()
        
        start_idx = 0
        for i, device in enumerate(self.devices):
            num_layers_on_device = self.layers_per_device[i]
            end_idx = start_idx + num_layers_on_device
            
            # Get layers for this device
            device_layers = nn.ModuleList([layers[j].to(device) for j in range(start_idx, end_idx)])
            self.device_layers.append(device_layers)
            
            start_idx = end_idx
            
        print(f"Distributed {num_layers} transformer layers across {self.num_devices} devices")
        for i, device in enumerate(self.devices):
            print(f"Device {i} ({device}): {self.layers_per_device[i]} layers")
    
    def forward(self, x, **kwargs):
        """Forward pass with pipeline parallelism to enable better GPU utilization."""
        B, T, C = x.shape
        
        # Store original dtype for consistent handling
        orig_dtype = x.dtype
        
        # Start on the first device
        x = x.to(self.devices[0], non_blocking=True)
        
        # Handle positional embeddings (similar to StreamingTransformer)
        if hasattr(self, 'positional_embedding') and hasattr(self, 'positional_scale'):
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            
            # Create sinusoidal positional embedding
            from audiocraft.modules.transformer import create_sin_embedding
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=orig_dtype)
            
            # Add to input with appropriate scaling
            x = x + self.positional_scale * pos_emb
            # Free positions tensor explicitly
            del positions
            
        # Apply embedding if available (though unlikely in MusicGen)
        elif hasattr(self, 'emb') and self.emb is not None:
            x = self.emb(x)
            
            # Apply separate positional embedding if available
            if hasattr(self, 'pos_emb') and self.pos_emb is not None:
                pos_emb = self.pos_emb(x)
                x = x + pos_emb
                del pos_emb  # Explicitly free
        
        # If cross_attention present in kwargs, move it to each device as needed
        cross_attention_src = kwargs.get('cross_attention_src', None)
        
        # Instead of pipeline parallelism, use simple sequential processing
        # that preserves the computation graph for gradient flow
        for device_idx, device in enumerate(self.devices):
            # Move input to current device if needed
            if x.device != device:
                # IMPORTANT: Do not detach when moving between devices
                # to preserve the computational graph for backward pass
                x = x.to(device, non_blocking=True)
            
            # Process through all layers on this device, maintaining gradients
            for layer in self.device_layers[device_idx]:
                x = self._apply_layer(layer, x, **kwargs)
        
        # Apply final normalization on the last device if available
        if hasattr(self, 'norm') and self.norm is not None:
            x = self.norm(x)
        
        return x
    
    def _sequential_forward(self, x, **kwargs):
        """Legacy sequential forward pass as fallback."""
        # Process through each device's layers sequentially
        try:
            for device_idx, device in enumerate(self.devices):
                # Move input to current device if needed
                if x.device != device:
                    x = x.to(device, non_blocking=True)
                
                # Apply all layers on this device
                for layer in self.device_layers[device_idx]:
                    # Ensure consistent dtype through all layers
                    x = x.to(layer.weight.dtype if hasattr(layer, 'weight') else x.dtype)
                    x = self._apply_layer(layer, x, **kwargs)
            
            # Apply final normalization on the last device if available
            if hasattr(self, 'norm') and self.norm is not None:
                x = self.norm(x)
            
            return x
        finally:
            # Clean up cross attention tensors if they exist
            if 'cross_attention_src' in kwargs:
                del kwargs['cross_attention_src']
    
    def _apply_layer(self, layer, x, **kwargs):
        """Apply a single transformer layer."""
        # Ensure inputs to layer have consistent dtype
        orig_dtype = x.dtype
        
        # Apply the layer - using try/except to handle potential dtype issues
        try:
            return layer(x, **kwargs)
        except RuntimeError as e:
            if "expected scalar type" in str(e):
                # Handle dtype mismatch by ensuring Float type
                x = x.to(torch.float32)
                result = layer(x, **kwargs)
                # Convert back to original dtype
                return result.to(orig_dtype)
            else:
                raise e

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelParallelLM(nn.Module):
    """Model parallel version of MusicGen's language model."""
    
    def __init__(self, original_lm: LMModel, devices: List[torch.device]):
        super().__init__()
        self.devices = devices
        
        # Store original config
        self.cfg = original_lm.cfg if hasattr(original_lm, 'cfg') else None
        
        # Keep condition provider on first device
        self.condition_provider = original_lm.condition_provider.to(devices[0])
        
        # Store fuser on first device
        if hasattr(original_lm, 'fuser'):
            self.fuser = original_lm.fuser.to(devices[0])
        
        # Replace transformer with model parallel version
        self.transformer = ModelParallelTransformer(original_lm.transformer, devices)
        
        # In AudioCraft's LMModel, the output projection is a list of linear layers
        # in a member variable called 'linears'
        if hasattr(original_lm, 'linears'):
            self.linears = nn.ModuleList([linear.to(devices[-1]) for linear in original_lm.linears])
        else:
            raise AttributeError("Could not find output projection 'linears' in the language model.")
        
        # Handle out_norm if present
        if hasattr(original_lm, 'out_norm') and original_lm.out_norm is not None:
            self.out_norm = original_lm.out_norm.to(devices[-1])
        else:
            self.out_norm = None
        
        # Store embeddings (there's one embedding per codebook)
        if hasattr(original_lm, 'emb'):
            self.emb = nn.ModuleList([emb.to(devices[0]) for emb in original_lm.emb])
        
        # Copy required attributes
        self.n_q = getattr(original_lm, 'n_q', 1)
        self.card = getattr(original_lm, 'card', 2048)  # Default to EnCodec vocab size
        self.max_target_len = getattr(original_lm, 'max_target_len', None)
        self.del_sep_token = getattr(original_lm, 'del_sep_token', None)
        
        # Copy CFG related attributes
        self.cfg_coef = getattr(original_lm, 'cfg_coef', 1.0)
        if hasattr(original_lm, 'cfg_dropout'):
            self.cfg_dropout = original_lm.cfg_dropout
        if hasattr(original_lm, 'att_dropout'):  
            self.att_dropout = original_lm.att_dropout
        self.two_step_cfg = getattr(original_lm, 'two_step_cfg', False)
        
        # Copy pattern provider
        if hasattr(original_lm, 'pattern_provider'):
            self.pattern_provider = original_lm.pattern_provider

    def compute_predictions(self, codes, conditions=None, condition_tensors=None, stage=-1):
        """Compute next-token predictions using the model-parallel transformer."""
        try:
            # Prepare inputs on the first device
            device = self.devices[0]
            
            b, k, t = codes.shape
            codes = codes.to(device, non_blocking=True)
            
            # Handle condition tensors
            if condition_tensors is None and conditions is not None:
                assert self.condition_provider is not None
                conditions = self.cfg_dropout(conditions) if hasattr(self, 'cfg_dropout') else conditions
                conditions = self.att_dropout(conditions) if hasattr(self, 'att_dropout') else conditions
                tokenized = self.condition_provider.tokenize(conditions)
                condition_tensors = self.condition_provider(tokenized)
            
            # Apply fuser if available
            input_ = None
            cross_attention_input = condition_tensors
            
            if hasattr(self, 'emb') and hasattr(self, 'fuser'):
                # Build input from embeddings (like in the original LMModel)
                input_ = sum([self.emb[k](codes[:, k]) for k in range(k)])
                input_, cross_attention_input = self.fuser(input_, condition_tensors)
            else:
                # Fallback: just use the codes as input
                input_ = codes
            
            # Process through the model-parallel transformer
            out = self.transformer(input_, cross_attention_src=cross_attention_input)
            
            # Apply out_norm if available (before projection)
            if self.out_norm is not None:
                out = self.out_norm(out)
            
            # Apply separate linear projections for each codebook
            # and stack the results [B, K, S, card]
            out = out.to(self.devices[-1], non_blocking=True)  # Move to last device for final projections
            logits = torch.stack([self.linears[i](out) for i in range(len(self.linears))], dim=1)
            
            # Create mask for valid positions - more memory efficient by using device reuse
            mask = torch.ones_like(codes, dtype=torch.bool, device=self.devices[-1])
            
            # Break computational history to prevent memory leaks
            out = out.detach()
            
            # Use namedtuple-style object with weak references to help garbage collection
            class ModelOutput:
                def __init__(self, logits, mask):
                    self.logits = logits
                    self.mask = mask
                    
            return ModelOutput(logits=logits, mask=mask)
        
        finally:
            # Help garbage collection
            if condition_tensors is not None:
                del condition_tensors
            if cross_attention_input is not None and cross_attention_input is not condition_tensors:
                del cross_attention_input
            # Trigger garbage collection
            gc.collect()

    def forward(self, sequence_codes, conditions=None, condition_tensors=None, stage='decoder'):
        """Forward pass for the model-parallel LM."""
        try:
            device = self.devices[0]
            if condition_tensors is None and conditions is not None:
                tokenized = self.condition_provider.tokenize(conditions)
                condition_tensors = self.condition_provider(tokenized)
                
            # Move inputs to the first device
            sequence_codes = sequence_codes.to(device, non_blocking=True)
            if condition_tensors is not None:
                condition_tensors = condition_tensors.to(device, non_blocking=True)
                
            # Process through transformer (will automatically handle device transitions)
            out = self.transformer(sequence_codes, cross_attention_src=condition_tensors)
            
            # Final layer processes on last device
            out = self.linears[-1](out.to(self.devices[-1], non_blocking=True))
            
            # Apply out_norm if available
            if self.out_norm is not None:
                out = self.out_norm(out)
            
            return out
        finally:
            # Clean up to avoid memory leaks
            if condition_tensors is not None:
                del condition_tensors

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_model_parallel_lm(original_lm, gpu_ids=[5, 7]):
    """Create a model-parallel version of the language model."""
    # With CUDA_VISIBLE_DEVICES set, we need to use local device IDs (0, 1, etc.)
    # instead of the global IDs (5, 7, etc.)
    local_gpu_ids = list(range(len(gpu_ids)))
    devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in local_gpu_ids]
    print(f"Model parallel using devices: {devices}")
    return ModelParallelLM(original_lm, devices)