# MuXiT
This is the repository for the FYP of 2024-25 Cohort, supervised by Prof. Andrew Horner. Group code is HO3.

# Front-end files
- For prototyping purposes, please modify ```Gradio.py```.

## Changelogs:
- Get new script for root
  - `npm run start` to start the whole frontend and backend system
  - `npm run start-backend` to awake the backend 
  - `npm run start-frontend` to awake the frontend 
  - `npm run build` to refresh and build frontend when initialising on a new env, error invoked, or updates
- Hosts
  - Frontend hosted at localhost:3000 port
  - Backend hosted at localhost:8000 port (Experimental: Gemma-3 (1B) chatbot, when deployed, runs on localhost:8001)
- Debugging
  - Fixed 422 error for the FASTAPI presentation problem and getting to test for the connection with model
  - Fixed CORS error using middleware
  - Fixed dark mode text

## Updates on 20250422:
- Added duration parameter
- Added local storage system, users can keep the history even after closing the website
- Bugs:
  - Axios is not able to be resolved, better to add resolver on webpack when working on the miniLLM
  - The user message disappeared right after sending, should be bug related to the local storage, I will try to fix it, yet if anyone want help also can try...

Tomy Kwong | Crystal Chan
