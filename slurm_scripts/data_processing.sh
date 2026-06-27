#!/bin/bash

# Do this if on server first
# Load anaconda environment with the requirements.txt
module load anaconda3/2022.10
conda activate dataviz_env

# Use this command to run preprocess file 
python data_processing.py