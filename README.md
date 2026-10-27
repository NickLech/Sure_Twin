# Setup & Installation Guide

## Prerequisites & Requirements

* **Operating System:** Tested on Ubuntu 26.04 LTS
* **Python:** Version `3.11.x`
* **System Packages:** `git`, `python3-venv`, `python3-pip`, `curl`, `unzip`
* **Hardware:** Tested on CPU (stay tuned for the CUDA GPU version)

---

## Option 1: Automatic Installation (Recommended)

An automated script `setup.sh` is provided to install system dependencies, set up `uv`, create the virtual environment, and install all Python packages listed in `requirements.txt`.

1. **Make the script executable:**
   ```bash
   chmod +x setup.sh
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

---

## Option 2: Manual Installation

If you prefer to configure your environment step-by-step, follow the instructions below:

### 1. Install System Dependencies
Update package lists and install required tools:
```bash
apt update
apt install -y git python3-venv python3-pip curl unzip
```

### 2. Install `uv`
Install `uv` to manage Python versions and virtual environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```
*(Note: You may need to restart your terminal or re-source your profile for `uv` to be available in your `PATH`.)*

### 3. Create & Activate Virtual Environment
Create a Python 3.11 virtual environment named `.venv` inside the project directory and activate it:
```bash
mkdir -p ~/SURE-Twin/datasets
cd ~/SURE-Twin
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install PyTorch (CPU version)
uv pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu

# Install BasicTS
uv pip install basicts
```

---

## Verify

To verify that the installation succeeded and the virtual environment is working properly, run:

```bash
python --version
python -c "import torch, basicts; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```text
Python 3.11.x
PyTorch version: 2.12.1 | CUDA available: False
```
*(If `CUDA available` returns `False`, PyTorch is correctly configured to use the CPU.)*
