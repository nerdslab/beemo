#!/usr/bin/env bash

VENV_DIR=${VENV_DIR:-venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

# If first argument is "y", skip prompt
SKIP_PROMPT=0
if [[ "${1:-}" == "y" ]]; then
  SKIP_PROMPT=1
fi

echo "VENV_DIR: ${VENV_DIR}"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
if [[ $SKIP_PROMPT -ne 1 ]]; then
  read -p "Press enter to continue or Ctrl+C to cancel..."
  echo
fi

uv venv $VENV_DIR -p $PYTHON_VERSION
source $VENV_DIR/bin/activate

uv pip install --upgrade pip
uv pip install -r requirements.txt

# Install xformers (handle GPU compatibility)
echo
echo "Installing xformers..."
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown")
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
# pip install xformers==0.0.32.post2

# if [[ "$GPU_MODEL" == *"5090"* ]]; then
#     echo "Detected 5090 GPU — installing xformers from source."
#     pip install "git+https://github.com/facebookresearch/xformers.git@5146f2a#egg=xformers"
# else
#     echo "Installing stable xformers version..."
#     pip install xformers==0.0.32.post2 || {
#         echo "Stable xformers failed — falling back to source install."
#         pip install "git+https://github.com/facebookresearch/xformers.git@5146f2a#egg=xformers"
#     }
# fi

echo
echo "✅ Environment setup complete!"