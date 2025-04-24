import os
import sys
import shutil
import numpy as np
import soundfile as sf

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from clap_score import CLAPScore

# 630k-audioset (for general audio less than 10-sec)
SAMPLE_RATE = 48000
LENGTH_IN_SECONDS = 30

# init
clap = CLAPScore(
    ckpt_dir= None,
    submodel_name="630k-audioset",
    verbose=True,
    audio_load_worker=8,
    enable_fusion=True,
)

#clap_score = clap.score(
#    text_path="./old_spec.csv",
#    audio_dir="./old_audio",
#    text_column="caption",
#    text_embds_path=None,
#    audio_embds_path=None,
#)
#print(f"Old CLAP score text and audio matching [mu. std]: {clap_score}")

clap_score = clap.score(
    text_path="./EF/musicgen_EF.csv",
    audio_dir="./EF/MusicGen",
    text_column="caption",
    text_embds_path=None,
    audio_embds_path=None,
)
print(f"MusicGen CLAP score text and audio matching [mu. std]: {clap_score}")

clap_score = clap.score(
        text_path="./EF/ours_EF.csv",
        audio_dir="./EF/Ours",
        text_column="caption",
        text_embds_path=None,
        audio_embds_path=None,
)
print(f"Ours CLAP score text and audio matching [mu. std]: {clap_score}")
