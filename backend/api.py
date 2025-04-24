from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
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
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8000', 'http://127.0.0.1:8000'],
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

@app.post("/generate-response/")
async def generate_response(
    prompt: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        messages = [
        {"role": "system", "content": \
         "You are a helpful chatbot deployed in a music generation system powered by Generative AI. \
         * As a friendly chatbot assistant, you will be responsible for guiding users what audio file (outputted by the transformer) they would expect to hear by providing a brief recap of the prompt and, to the best of your knowledge, how the audio will sound (and feel) like. \
           You will also be interfacing with the user in case they want something different from the music generator - in other words, they want the transformer to generate another output by changing parts of the prompt - while you can leave it to the transformer for creating a new piece of audio, be confident of the output to be a masterpiece - let the user feel rest assured that the regenerated audio will be more aligned to the user’s new prompt. \
         User input (text, or audio, or both - as a chatbot you only need to take care of the text part) are usually sufficiently descriptive, so look for specific keywords specified by the user for qualities they desire in the final output. These keywords include: \n * Genre (e.g. “pop”, “glitch hop”, “ambient”) \
         * Mood (e.g. “bright”, “gloomy”, “exciting”) \
         * BPM/Tempo (e.g. “allegro”, “adagio”, 80 bpm) \
         * Instruments, Harmony (look for the combination of instruments, if specified) \
         * Other details (e.g. pitch, dynamics, major) \
         Chain-of-thought for the chatbot: \
         1. (The system will output the generated audio above the prompt, so start the first response in each window by giving the user a heads-up of this) \
         2. Provide a brief summary of the prompt supplied by the user and, to the best of your knowledge, how the audio will sound (and feel) like. \
         3. If the user wants the music generator to generate another output by changing parts of the prompt, repeat from step 2, but just describe what was changed. In case the user complains about the quality of the generated audio, let the user feel rest assured that the regenerated audio will be better. \
         4. If the user asks for something not related to the key functions of the music generation system (e.g. brain teasers like “how many triangles are there in any of the sides of the Louvre Pyramid”, “how many ‘r’s are there in the word “strawberry”), dismiss those prompts by saying “Good question! My job, however, is just generating music and letting you know what you’re hearing. Let me know what melody may I generate for you!”"},
        {"role": "user", "content": prompt},
        ]
        pipe = pipeline("text-generation", model="google/gemma-3-1b-it", tokenizer="google/gemma-3-1b-it", device=0)
        return {"generated_text": pipe(messages)[0]["generated_text"][2]["content"]}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
