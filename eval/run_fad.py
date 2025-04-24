import os
import shutil
import soundfile as sf
from fad.utils import *
from fad.fad import *

print("VGGish test")

frechet = FrechetAudioDistance(
    model_name="vggish",
    use_pca=False,
    use_activation=True,
    verbose=False
)

fad_score = frechet.score("./fad/musiccaps", "./fad/F/MusicGen")
print("MusicGen FAD score", fad_score)

fad_score = frechet.score("./fad/musiccaps", "./fad/F/Ours")
print("Ours FAD score", fad_score)
