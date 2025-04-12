import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set the GPU device to use
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from peft import PeftModel

def load_lora_model(model_id, lora_weights_path):
    """Load a MusicGen model with LoRA weights."""
    print(f"Loading base model: {model_id}")
    base_model = MusicGen.get_pretrained(model_id)
    
    print(f"Applying LoRA weights from: {lora_weights_path}")
    if os.path.isdir(lora_weights_path):
        # Directory format (PEFT's save_pretrained)
        base_model.lm = PeftModel.from_pretrained(base_model.lm, lora_weights_path)
    else:
        # File format (our custom format)
        state_dict = torch.load(lora_weights_path, map_location='cpu')
        if "lora_state_dict" in state_dict:
            base_model.lm.load_state_dict(state_dict["lora_state_dict"], strict=False)
    
    return base_model

def generate_with_lora(
    model_id: str = "facebook/musicgen-small",
    lora_path: str = "models/lora/musicgen_lora_final.pt",
    prompts: list = None,
    output_dir: str = "outputs",
    duration: float = 10.0,
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    use_sampling: bool = True,
    seed: int = None
):
    """Generate audio using a MusicGen model fine-tuned with LoRA."""
    # Set seed if specified
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Load model with LoRA weights
    model = load_lora_model(model_id, lora_path)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
        use_sampling=use_sampling
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default prompts if none provided
    if prompts is None:
        prompts = ["A beautiful piano melody with soft strings in the background",
                   "Energetic electronic dance music with a strong beat and synthesizer"]
    
    # Generate audio
    print(f"Generating {len(prompts)} samples...")
    generated_audio = model.generate(prompts)
    
    # Save audio files
    print(f"Saving audio to {output_dir}")
    for idx, (audio, prompt) in enumerate(zip(generated_audio, prompts)):
        filename = f"{idx}_lora_sample"
        audio_write(
            os.path.join(output_dir, filename),
            audio.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
            metadata={"prompt": prompt}
        )
        print(f"Saved {filename}.wav")
    
    print("Generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio with a LoRA-fine-tuned MusicGen model")
    parser.add_argument("--model_id", type=str, default="facebook/musicgen-small", 
                        help="Base model ID (e.g., facebook/musicgen-small)")
    parser.add_argument("--lora_path", type=str, required=True, 
                        help="Path to LoRA weights file or directory")
    parser.add_argument("--prompts", nargs="+", default=None, 
                        help="Text prompts for generation (multiple allowed)")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save generated audio")
    parser.add_argument("--duration", type=float, default=10.0, 
                        help="Duration of generated audio in seconds")
    parser.add_argument("--top_k", type=int, default=250, 
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.0, 
                        help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Sampling temperature")
    parser.add_argument("--cfg_coef", type=float, default=3.0, 
                        help="Classifier-free guidance coefficient")
    parser.add_argument("--no_sampling", action="store_false", dest="use_sampling", 
                        help="Disable sampling (use argmax)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_with_lora(
        model_id=args.model_id,
        lora_path=args.lora_path,
        prompts=args.prompts,
        output_dir=args.output_dir,
        duration=args.duration,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
        use_sampling=args.use_sampling,
        seed=args.seed
    )