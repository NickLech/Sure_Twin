# University of Trento GPU Cluster Guide

This guide details the steps required to configure the environment, install CUDA-compatible packages, and submit training jobs on the University of Trento GPU cluster (`baldo`) using SLURM.

---

## Prerequisites & Requirements

* **Platform:** Tested on University of Trento GPU Cluster `baldo`, with `SLURM` scheduler
* **Python:** Tested on version `3.9.21`
* **PyTorch CUDA Target:** Tested on CUDA 12.4 (`cu124`)
* **System Packages:** `git`, `python3-venv`, `python3-pip`, `curl`, `unzip`

---

## 1. Verify SLURM Account

Before submitting jobs, check your assigned account, partition, and QoS settings:

```bash
sacctmgr show assoc user=$USER format=account%40,partition,qos%20
```

---

## 2. Virtual Environment Setup

Activate the project's virtual environment and verify `pip` availability inside the venv:

```bash
source ~/Sure_Twin/.venv/bin/activate
python -m pip --version
```

If `pip` is missing or needs an update inside the virtual environment, run:

```bash
python -m ensurepip --upgrade
```

> ⚠️ **Important:** Always use `python -m pip` instead of `pip` directly. Using `pip` alone may invoke the system Python 3.9 and install packages into `~/.local` instead of your `.venv`.

---

## 3. Install PyTorch for CUDA 12.4

The cluster GPU driver supports up to CUDA 12.4. Remove any incompatible builds and install the matching PyTorch release (~2.5 GB download):

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verify PyTorch Installation

Run the following inline command to check the PyTorch build and CUDA version:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**Expected output:**
```text
2.6.0+cu124 12.4
```

---

## 4. Quick GPU Verification Job

Test GPU access on the cluster by launching a short interactive/wrapper job on the `edu-short` partition:

```bash
sbatch -N 1 -n 1 -p edu-short --time=00:05:00 -A fabrizio.granelli.edu --qos=gpuedu --gres=gpu:1 --cpus-per-task=8 --mem=32G --output=test_%j.log --wrap="source ~/Sure_Twin/.venv/bin/activate; python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'"
```

Check job status and verify the log output:

```bash
squeue -u $USER
cat test_<job_id>.log
```

**Expected output:**
```text
True 12.4
```

---

## 5. Job Script Configuration & Execution

Create or update your batch submission script `job.sh` in the project root:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p edu-medium
#SBATCH -A fabrizio.granelli.edu
#SBATCH --qos=gpuedu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=out_%j.log

source ~/Sure_Twin/.venv/bin/activate
cd ~/Sure_Twin

python train.py
```

### SLURM Management Commands

* **Submit job:**
  ```bash
  sbatch job.sh
  ```
* **Check queue status:**
  ```bash
  squeue -u $USER
  ```
* **Monitor log output in real time:**
  ```bash
  tail -f out_<job_id>.log
  ```
* **Cancel a running/queued job:**
  ```bash
  scancel <job_id>
  ```

---

## Partition Reference

| Partition | Max Time | Usage Notes |
| :--- | :--- | :--- |
| `edu-short` | 5 minutes | Quick testing & verification |
| `edu-medium` | 2 hours | Standard training runs |
| `edu-long` | 24 hours | Extended training (change `-p edu-long` and `--time`) |

---

## Troubleshooting & Cleanup

If packages were accidentally installed globally via system `pip` (in `~/.local`), clean up home disk space with:

```bash
# Check occupied space in ~/.local
du -sh ~/.local

# Remove system-level PyTorch packages (Python 3.9)
pip uninstall -y torch torchvision torchaudio
```