#!/bin/bash

set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "uv installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "venv creation..."
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "Python dependencies installation..."
uv pip install -r requirements.txt

echo "Verifying..."
python --version
python -c "import torch, basicts; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "Setup successfully completed."
echo "Note: CUDA available may be False on the login node (no GPU). Check it again on a GPU node via srun."
