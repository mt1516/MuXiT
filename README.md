# MuXiT
This is the repository for the FYP of 2024-25 Cohort, supervised by Prof. Andrew Horner. Group code is HO3.

# Training Data Description
The dataset used is [FMA](https://os.unil.cloud.switch.ch/fma/fma_full.zip) (Defferrard, Benzi, Vandergheynst, and Bresson, 2017), which, in full, features 106,574 soundtracks (of full length) spanning across 161 genres. Downloading the dataset using the link to the left allows access to all metadata files and soundtracks (specifically, 17 out of 156 folders of soundtracks - randomly sampled - are used to optimise storage).

Data cleaning procedure:
0. Identify useful information from metadata (tracks.csv, found in the FMA zip file) (See comments in ```txtGen.py``` for description of useful fields)
1. Generate (trackID).txt by running ```txtGen.py```
2. Aggregate all .txt files (generated in 1.) into NewTracks.csv (or AggTracks.csv) by running ``csvAgg.py```
3. Generate tracks.json from NewTracks.csv (or AggTracks.csv) by running ```jsonify.py```
