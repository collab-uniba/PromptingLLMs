#!/bin/bash

# Check if model name argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name="$1"

source venv/bin/activate

# Update the model name in the configuration file
python change_model.py "$model_name"

# Update the output and error file names in the SLURM script
python update_slurm.py "$model_name"

# Download the model
./download_model.sh "$model_name"
