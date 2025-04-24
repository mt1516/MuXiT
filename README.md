![image](https://github.com/user-attachments/assets/ad10d2e8-732a-4dee-bda4-5268ec9c3196)![image](https://github.com/user-attachments/assets/13b116cd-fec9-459c-b640-b06c08cfe691)# MuXiT
This is the repository for the FYP of 2024-25 Cohort, supervised by Prof. Andrew Horner. Group code is HO3.

# Training Data Description
The dataset used is [FMA](https://os.unil.cloud.switch.ch/fma/fma_full.zip) (Defferrard, Benzi, Vandergheynst, and Bresson, 2017), which, in full, features 106,574 soundtracks (of full length) spanning across 161 genres. Downloading the dataset using the link to the left allows access to all metadata files and soundtracks (specifically, 17 out of 156 folders of soundtracks - randomly sampled - are used to optimise storage).

Data cleaning procedure:

0. Identify useful information from metadata (tracks.csv, found in the FMA zip file) (See comments in ```txtGen.py``` for description of useful fields)
1. Generate (trackID).txt by running ```txtGen.py```
2. Aggregate all .txt files (generated in 1.) into NewTracks.csv (or AggTracks.csv) by running ```csvAgg.py```
3. Generate tracks.json from NewTracks.csv (or AggTracks.csv) by running ```jsonify.py```

# System Description
The chronology of branch development is summarised as follows:
front-end-dev → back-end-dev & detached / experimental_multi_GPU (model training) → JS_frontend → front-end (where the core components of the system reside, including the backend)

## Backend Description
[Update: Contents in this branch have been merged with the frontend]
This branch houses the backend scripts that host the music generator model inference, as well as the SLM module. Highlights:
- ```api.py```: Scripts that serve the required backend modules
- ```inference_class.py```: Inference class definition for the music generator model

## Frontend Description
This branch houses the frontend scripts that host the Next.js site on which the user interface of the system runs. Highlights:
- ```Gradio.py```: For early prototyping purposes.
- Hosts:
-- Frontend hosted at localhost:3000 (127.0.0.1:3000)
-- Backend hosted at localhost:8000 (127.0.0.1:8000)
- Running the system:
-- ```npm run start``` to start the whole frontend and backend system (For best compatibility, execute this command in the ```(MuXiT\)jsfrontend``` directory)
-- ```npm run start-backend``` to awake the backend (Alternatively, run ```backend\api.py``` in another terminal window on the same machine)
-- ```npm run start-frontend``` to awake the frontend
- System features spotlight:
-- Local chat history: Keep your past chats (all text and audio files), even after you have closed the server!
-- Customising music generation: On top of text prompts, feel free to upload audio clips to generate more creative stuff!
-- SLM integration: Get friendly responses with every message sent in the system! Powered by Google Gemma 3
