# Setup & Installation Guide

## Prerequisites & Requirements

* **Platform:** tested on the University of Trento GPU cluster `baldo` (SLURM scheduler)
* **Python:** 3.11 (the virtual environment is created with `uv`, see below)
* **PyTorch CUDA target:** CUDA 12.4 (`cu124`)
* **System packages:** `git`, `curl`, `unzip`

> The system Python on the cluster is older (3.9). Always work inside the project's `.venv`, never with the system interpreter.

---

## Option 1: Automatic installation (recommended)

`scripts/setup.sh` checks/installs `uv`, creates the virtual environment `.venv` with Python 3.11 and installs the pinned packages listed in `requirements.lock`.

1. **Make the script executable:**
   ```bash
   chmod +x scripts/setup.sh
   ```

2. **Run it from the repository root:**
   ```bash
   ./scripts/setup.sh
   ```

3. **Activate the virtual environment:**
   *(The script runs in a subshell, so the environment must be activated manually afterwards.)*
   ```bash
   source .venv/bin/activate
   ```

---

## Option 2: Manual installation

### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```
*(Restart the terminal or re-source your profile if `uv` is not recognized immediately.)*

### 2. Create and activate the virtual environment
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 3. Install the dependencies
```bash
uv pip install -r requirements.lock --index-strategy unsafe-best-match
```

`requirements.txt` lists the direct dependencies; `requirements.lock` pins the full environment (including `torch==2.6.0+cu124`) and is the file to use for a reproducible install.

---

## Verify

With `.venv` active:

```bash
python --version
python -c "import torch, basicts; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```text
Python 3.11.x
PyTorch version: 2.6.0+cu124 | CUDA available: <True/False>
```
*(`CUDA available` may be `False` on login nodes or CPU-only machines. On a GPU node it must be `True`.)*

---

## Execution

Once the environment is set up and activated, run everything from the repository root. For a quick single-seed test:

```bash
source .venv/bin/activate
python src/training/train_STID.py --seed 0
```

The full workflow (data conversion, ensemble training, evaluation, conformal bounds) is described in the [main README](../README.md#4-step-by-step-run).

> **Cluster deployment:** on the UniTrento SLURM GPU cluster (`baldo`) see the dedicated [Cluster Guide](CLUSTER_GUIDE.md).
