# Setup & Installation Guide

## Prerequisites & Requirements

* **Platform:** Tested on University of Trento GPU Cluster `baldo`, with `SLURM` scheduler
* **Python:** Tested on version `3.9.21`
* **PyTorch CUDA Target:** Tested on CUDA 12.4 (`cu124`)
* **System Packages:** `git`, `python3-venv`, `python3-pip`, `curl`, `unzip`

---

## Option 1: Automatic Installation (Recommended)

An automated script `setup.sh` is provided to check/install `uv`, create the virtual environment `.venv`, and install all Python packages listed in `requirements.txt`.

1. **Make the script executable:**
   ```bash
   chmod +x setup.sh
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Activate the virtual environment:**
   *(Note: Script execution ends in a subshell, so you need to activate the environment manually afterwards)*
   ```bash
   source .venv/bin/activate
   ```

---

## Option 2: Manual Installation

If you are excited to configure your environment step-by-step, follow the instructions below:

### 1. Install `uv`
Install `uv` to manage Python versions and virtual environments:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```
*(Note: Restart your terminal or re-source your profile if `uv` is not recognized immediately.)*

### 2. Create & Activate Virtual Environment
Create a Python 3.11 virtual environment named `.venv` in the project root and activate it:
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

---

## Verify

To verify that the installation succeeded and the virtual environment is working properly, activate `.venv` and run:

```bash
python --version
python -c "import torch, basicts; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```text
Python 3.11.x
PyTorch version: <version> | CUDA available: <True/False>
```
*(Note: `CUDA available` may return `False` on login nodes or CPU-only machines. On a GPU node, it should return `True`.)*

---

## Execution

Once the environment is set up and activated:

```bash
source .venv/bin/activate
python train.py
```

> 💡 **Cluster Deployment:** If you are running on the UniTrento SLURM GPU cluster (`baldo`), please refer to the dedicated [Cluster Guide](CLUSTER_GUIDE.md).
