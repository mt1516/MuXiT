# MuXiT
This is the repository for the FYP of 2024-25 Cohort, supervised by Prof. Andrew Horner. Group code is HO3.

# System Description
The chronology of branch development, in parallel of the main branch, is summarised as follows:
front-end-dev → back-end-dev & detached / experimental_multi_GPU (model training) → JS_frontend → front-end (where the core components of the system reside, including the backend)

# Training Data Description
[Main contributor: Tomy Kwong]
The dataset used is [FMA](https://os.unil.cloud.switch.ch/fma/fma_full.zip) (Defferrard, Benzi, Vandergheynst, and Bresson, 2017), which, in full, features 106,574 soundtracks (of full length) spanning across 161 genres. Downloading the dataset using the link to the left allows access to all metadata files and soundtracks (specifically, 17 out of 156 folders of soundtracks - randomly sampled - are used to optimise storage).

Data cleaning procedure:

0. Identify useful information from metadata (tracks.csv, found in the FMA zip file) (See comments in ```txtGen.py``` for description of useful fields)
1. Generate (trackID).txt by running ```txtGen.py```
2. Aggregate all .txt files (generated in 1.) into NewTracks.csv (or AggTracks.csv) by running ```csvAgg.py```
3. Generate tracks.json from NewTracks.csv (or AggTracks.csv) by running ```jsonify.py```

## Backend Description
[Update: Contents in this branch have been merged with the frontend]
[Main contributor: Eric Kwok]
This branch houses the backend scripts that host the music generator model inference, as well as the SLM module. Highlights:
- ```api.py```: Scripts that serve the required backend modules
- ```inference_class.py```: Inference class definition for the music generator model
Before proceeding to the backend, please make sure all Python library dependencies are collected by running ```(python -m) pip install -r (dependencies.txt or requirements.txt)```. This .txt file is located in the ```(MuXiT\)backend``` directory.

## Frontend Description
[Main contributors: Crystal Chan, Tomy Kwong]
This branch houses the frontend scripts that host the Next.js site on which the user interface of the system runs. Highlights:
- ```Gradio.py```: For early prototyping purposes.
- Hosts:
 - Frontend hosted at localhost:3000 (127.0.0.1:3000)
 - Backend hosted at localhost:8000 (127.0.0.1:8000)
- Running the system:
 - ```npm run start``` to start the whole frontend and backend system (For best compatibility, execute this command in the ```(MuXiT\)jsfrontend``` directory)
 - ```npm run start-backend``` to awake the backend (Alternatively, run ```backend\api.py``` in another terminal window on the same machine)
 - ```npm run start-frontend``` to awake the frontend
- System features spotlight:
 - Local chat history: Keep your past chats (all text and audio files), even after you have closed the server!
 - Customising music generation: On top of text prompts, feel free to upload audio clips to generate more creative stuff!
 - SLM integration: Get friendly responses with every message sent in the system! Powered by Google Gemma 3

## Model Training (detached / experimental_multi_GPU) Description
[Main contributor: Melvin Tong]
We performed LoRA (Low-Rank Adaptation) training on the CSE server. Training code can be found in the ```musicgen_trainer``` folder (courtesy of [@chavinlo](https://github.com/chavinlo/musicgen_trainer)). Other files on these branches are mostly log files produced in the output.
> During the training process, the pre-trained model was loaded and all components were explicitly converted to float32 precision to ensure numerical stability.
> The transformer layers were evenly partitioned across four GPUs, with each device responsible for twelve out of forty-eight layers.
> LoRA adapters were selectively injected into key linear submodules (linear1, linear2, and out_proj), resulting in approximately 28M trainable parameters —representing only 2.8% of the total model parameters.
