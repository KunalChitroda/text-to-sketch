#!/bin/bash

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running preprocessing..."
python data_preprocessing.py

echo "Starting training..."
python train_model.py

echo "Generating sketch..."
python generate_sketch.py --prompt "2BHK apartment with 1 kitchen and 1 bathroom"

echo "Visualizing output..."
python visualize_output.py
