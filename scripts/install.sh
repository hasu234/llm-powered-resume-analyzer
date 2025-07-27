#! /bin/bash

# Create virtual env using uv
uv venv

# Activate venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Proper PyTorch setup
# uv pip uninstall torch
# For GPU installation uncomment the following line 
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
