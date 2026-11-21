# University of Trento GPU Cluster Guide

This guide details the steps required to configure the environment, check CUDA-compatible packages, and submit training jobs on the University of Trento GPU cluster (`baldo`) using SLURM.

---

## Prerequisites & Requirements

* **Platform:** University of Trento GPU cluster `baldo`, `SLURM` scheduler
* **Python:** 3.11 inside the project's `.venv` (the system Python is 3.9: do not use it)
* **PyTorch CUDA target:** CUDA 12.4 (`cu124`)
* **System packages:** `git`, `curl`, `unzip`

The environment itself is created by `scripts/setup.sh` (see [SETUP.md](SETUP.md)). This guide covers what is specific to the cluster.

---

## 1. Verify your SLURM account

Before submitting jobs, check your assigned account, partition and QoS:

```bash
sacctmgr show assoc user=$USER format=account%40,partition,qos%20
```

The job scripts in `scripts/slurm/` contain the account and QoS used by the authors (`#SBATCH -A ...`, `--qos=gpuedu`). **Replace them with your own** before submitting.

---

## 2. Virtual environment

Activate the project's virtual environment and verify that `pip` points inside it:

```bash
source ~/Sure_Twin/.venv/bin/activate
python -m pip --version
```

If `pip` is missing or needs an update inside the virtual environment:

```bash
python -m ensurepip --upgrade
```

> **Important:** always use `python -m pip` instead of `pip`. Using `pip` alone may invoke the system Python 3.9 and install packages into `~/.local` instead of your `.venv`.

---

## 3. Check PyTorch for CUDA 12.4

The cluster GPU driver supports up to CUDA 12.4. `requirements.lock` already pins the matching build (`torch==2.6.0+cu124`), so no manual PyTorch installation is normally needed. Check it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**Expected output:**
```text
2.6.0+cu124 12.4
```

If you get an incompatible build, reinstall the matching release (about 2.5 GB download):

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Quick GPU verification job

Test GPU access with a short job on the `edu-short` partition (replace `<your-account>`):

```bash
sbatch -N 1 -n 1 -p edu-short --time=00:05:00 -A <your-account> --qos=gpuedu --gres=gpu:1 --cpus-per-task=8 --mem=32G --output=test_%j.log --wrap="source ~/Sure_Twin/.venv/bin/activate; python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'"
```

Check job status and the log:

```bash
squeue -u $USER
cat test_<job_id>.log
```

**Expected output:**
```text
True 12.4
```

---

## 5. Job scripts and execution

The batch scripts live in `scripts/slurm/`:

| Script | Purpose | Array | Time limit |
|---|---|---|---|
| `run_train.sh` | trains one STID per seed (`src/training/train_STID.py`) | `0-9` | 02:00:00 |
| `run_eval.sh` | evaluates each seed's best checkpoint on a split (`src/training/eval_split.py`) | `0-9` | 00:30:00 |

Both request one GPU on `edu-medium` and use the task id of the array as the seed. Excerpt of `run_train.sh`:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p edu-medium
#SBATCH -A <your-account>
#SBATCH --qos=gpuedu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=out_%A_%a.log

source ~/Sure_Twin/.venv/bin/activate
cd ~/Sure_Twin
python src/training/train_STID.py --seed $SLURM_ARRAY_TASK_ID
```

### SLURM management commands

* **Submit the training array:**
  ```bash
  sbatch scripts/slurm/run_train.sh
  ```
* **Submit the evaluation on the val split after training:**
  ```bash
  TRAIN=$(sbatch --parsable scripts/slurm/run_train.sh)
  sbatch --dependency=afterok:$TRAIN scripts/slurm/run_eval.sh val
  ```
* **Check queue status:**
  ```bash
  squeue -u $USER
  ```
* **Monitor a log in real time:**
  ```bash
  tail -f out_<job_id>_<seed>.log
  ```
* **Cancel a running/queued job:**
  ```bash
  scancel <job_id>
  ```

---

## Partition reference

| Partition | Max time | Usage notes |
| :--- | :--- | :--- |
| `edu-short` | 5 minutes | Quick testing & verification |
| `edu-medium` | 2 hours | Standard training runs |
| `edu-long` | 24 hours | Extended training (change `-p edu-long` and `--time`) |

---

## Troubleshooting & cleanup

If packages were accidentally installed globally via the system `pip` (in `~/.local`), clean up home disk space:

```bash
# Check occupied space in ~/.local
du -sh ~/.local

# Remove system-level PyTorch packages (system Python 3.9)
pip uninstall -y torch torchvision torchaudio
```
