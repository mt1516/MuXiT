import torchaudio
from tqdm import trange
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

import os
import typing as tp

import torch

from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel
from audiocraft.models.builders import get_debug_compression_model, get_debug_lm_model
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.data.audio_utils import convert_audio
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
from audiocraft.utils.autocast import TorchAutocast

class Inference:
    def __init__(self, prompt:str, duration:float=30, audio_input:str=None):
        self.prompt = prompt
        self.weights_path = '../../musicgen_lora_final.pt'
        self.save_path = 'output.wav'
        self.model_id = 'facebook/musicgen-stereo-melody'
        self.duration = duration
        self.sample_loops = 2
        self.use_sampling = True
        self.two_step_cfg = False
        self.top_k = 250
        self.top_p = 0.0
        self.temperature = 1.0
        self.cfg_coef = 3.0
        self.audio_input = audio_input
        self.frame_rate = 32_000

    def generate(self):
        model = MusicGen.get_pretrained(self.model_id)

        # print(self.lm.state_dict().keys())
        if self.weights_path is not None:
            #self.lm.load_state_dict(torch.load(self.weights_path), strict=True)
            model.lm.load_state_dict(torch.load(self.weights_path)['lora_state_dict'], strict=False)
        if self.audio_input is not None:
            print("melody")
            sample, sample_sr = torchaudio.load(self.audio_input)
            sample = convert_audio(sample, sample_sr, 32000, 2)
            wav = model.generate_with_chroma([self.prompt], sample, sample_sr)
            audio_write(self.save_path, wav[0].cpu(), 32000, strategy="loudness")
            
        else:
            print("No melody")
            attributes, prompt_tokens = model._prepare_tokens_and_attributes([self.prompt], None)
            print("attributes:", attributes)
            print("prompt_tokens:", prompt_tokens)

            duration = self.duration

            model.generation_params = {
                'max_gen_len': int(duration * self.frame_rate),
                'use_sampling': self.use_sampling,
                'temp': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'cfg_coef': self.cfg_coef,
                'two_step_cfg': self.two_step_cfg,
            }
            total = []
            for _ in trange(self.sample_loops):
                with model.autocast:
                    gen_tokens = model.lm.generate(prompt_tokens, attributes, callback=None, **model.generation_params)
                    total.append(gen_tokens[..., prompt_tokens.shape[-1] if prompt_tokens is not None else 0:])
                    prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2:]
            gen_tokens = torch.cat(total, -1)

            assert gen_tokens.dim() == 3
            print("gen_tokens information")
            print("Shape:", gen_tokens.shape)
            print("Dtype:", gen_tokens.dtype)
            print("Contents:", gen_tokens)
            with torch.no_grad():
                gen_audio = model.compression_model.decode(gen_tokens, None)
            print("gen_audio information")
            print("Shape:", gen_audio.shape)
            print("Dtype:", gen_audio.dtype)
            print("Contents:", gen_audio)
            gen_audio = gen_audio.cpu()
            torchaudio.save(self.save_path, gen_audio[0,:,0:duration*32000], model.sample_rate)
            #torchaudio.save(self.save_path, gen_audio[0], self.sample_rate)
