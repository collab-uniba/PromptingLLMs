#!/bin/bash

# Check if model_name argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name=$1

source venv/bin/activate

# Execute the Python script with the provided model_name
python3 download_model.py "$model_name"

