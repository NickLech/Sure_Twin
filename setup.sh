#!/bin/bash

set -e

echo "system packets installation..."
sudo apt update
sudo apt install -y git python3-venv python3-pip curl unzip

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "uv installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "venv creation..."
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "python dependencies installation..."
uv pip install -r requirements.txt

echo "verify..."
python --version
python -c "import torch, basicts; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "setup succesfully completed..."
