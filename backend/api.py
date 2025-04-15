from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import traceback
import shutil

from inference_class import Inference #import the inference class

app = FastAPI()

#add CORS middleware to accept http request from different sources
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicRequest(BaseModel):
    prompt: str
    duration: int  # Duration for each track

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@app.post("/generate-music/")
async def generate_music(request: MusicRequest, audio_input: UploadFile, background_tasks: BackgroundTasks):
    if request.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be greater than zero")
    try:
        if audio_input is not None:
            print("Saving audio input.")
            file_location = f"temp/audio_input.wav"
            os.makedirs(os.path.dirname(file_location), exist_ok=True)

            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(audio_input.file, buffer)
                print("Melody inference starts.")
                inference = Inference(prompt=request.prompt, duration=request.duration, audio_input="temp/audio_input.wav")

        else:
            print("No melody inference starts.")
            inference = Inference(prompt=request.prompt, duration=request.duration, audio_input=None)

        inference.generate() #output is 'output.wav'

        def iterfile():
            with open("output.wav", mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating music: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)