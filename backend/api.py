from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, File
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
#cors reject error
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:8001', 'http://127.0.0.1:8001'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-music/")
async def generate_music(
    prompt: str = Form(...),
    duration: int = Form(...),
    audio_input: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be greater than zero")
    try:
        if audio_input is not None:
            print("Saving audio input.")
            file_location = f"temp/audio_input.wav"
            os.makedirs(os.path.dirname(file_location), exist_ok=True)

            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(audio_input.file, buffer)
                print("Melody inference starts.")
                inference = Inference(prompt=prompt, duration=duration, audio_input="temp/audio_input.wav")
        else:
            print("No melody inference starts.")
            inference = Inference(prompt=prompt, duration=duration, audio_input=None)

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