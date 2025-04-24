import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

generation_result_path = "fad/kl_MusicGen_melody" #Change this
target_audio_path = "fad/background"

# Initialize a helper instance
evaluator = EvaluationHelper(32000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)

