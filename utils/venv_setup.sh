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

# torch and xformers come from the same CUDA 12.8 index so their builds match
# (torch on PyPI is built for CUDA 13)
echo
echo "Installing torch and xformers..."
uv pip install torch==2.11.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt

echo
echo "✅ Environment setup complete!"