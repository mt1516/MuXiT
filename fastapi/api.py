from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import scipy.io.wavfile
import torch
import random
import traceback

import audiocraft.models
from audiocraft.data.audio import audio_write


path_model = '/path/to/model/'
path_out = '/path/to/output/'

app = FastAPI()

class MusicRequest(BaseModel):
    prompt: str
    duration: int  # Duration for each track

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@app.post("/generate-music/")
async def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    if request.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be greater than zero")

    synthesiser = None

    try:
        # Set device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

        # Optionally limit GPU memory usage
        if device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, device=0)
                print("Limited GPU memory usage to 80%")
            except Exception as mem_error:
                print(f"Failed to limit GPU memory: {mem_error}")

        # Load MusicGen model
        musicgen = audiocraft.models.MusicGen.get_pretrained(path_model)
        musicgen.set_generation_params(duration=request.duration)

        print("Model loaded successfully")

        # Generate two audio tracks using a random seed
        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(random_seed)

        music1 = musicgen(request.prompt)
        random_seed += 1
        torch.manual_seed(random_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(random_seed)
        music2 = musicgen(request.prompt)

        # Save audio files
        output1 = os.path.join(path_out, "song1.wav")
        scipy.io.wavfile.write(output1, rate=music1["sampling_rate"], data=music1["audio"])
        #Output using MusicGen API audio_write
        #audio_write(output1, music1.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
        #FileResponse(path_out, media_type="audio/wav", filename="song1.wav")

        output2 = os.path.join(path_out, "song2.wav")
        scipy.io.wavfile.write(output2, rate=music2["sampling_rate"], data=music2["audio"])
        #Output using MusicGen API audio_write
        #audio_write(output2, music2.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
        #FileResponse(path_out, media_type="audio/wav", filename="song2.wav")

        #return {"song1": #FileResponse(path_out, media_type="audio/wav", filename="song1.wav"), "song2": #FileResponse(path_out, media_type="audio/wav", filename="song2.wav")}

        return {"song1": output1, "song2": output2}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating music: {e}")

    finally:
        if synthesiser:
            del synthesiser
        torch.cuda.empty_cache()
        print("Cleaned up resources")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)