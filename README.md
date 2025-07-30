# Diffusion Model Assignment – MNIST Generation

## Overview
In this assignment, you will **implement, train, and tune a diffusion model** to generate realistic MNIST digits.  
You will start from a **baseline UNet-based DDPM** we provide and can:
- **Tune hyperparameters** (learning rate, EMA decay, beta schedule, etc.).
- **Modify the model architecture** (swap UNet for Transformer, add residual blocks, change channels, etc.).

## What You Need to Do
1. **Train your diffusion model** using `train.py`.
2. **Optionally modify the model** in `models/model.py` (or create your own).
3. **Tune hyperparameters** by editing your config file (`configs/my_config.yaml`).
4. **Generate samples** using `sample.py`.
5. **Submit your results** (your checkpoint, config, and samples) so we can evaluate FID.

---

## Repository Structure
```

diffusion-mnist/
├── configs/                 # Config files for training
│   └── default.yaml         # Baseline config
├── data/                    
│   └── mnist/               # MNIST dataset will download here
├── models/                  # Model architectures
│   ├── model.py             # Main entry point (you can edit this)
│   └── unet.py              # Provided UNet baseline
├── outputs/                 # Checkpoints will be saved here 
├── train.py                 # Script to train a model
├── sample.py                # Script to generate samples from a trained model
├── evaluate.py              # Computes FID vs. MNIST test set
├── environment.yml          # Conda env yaml
└── README.md

````

---

## How to Use

### 1. Install Conda Envionment and Dependencies
We use Python 3.11 and PyTorch with CUDA 11.8.

```bash
pip install torch torchvision tqdm numpy scipy pyyaml
````

### 2. Train a Model

Use your own config file or the default one:

```bash
python train.py --config configs/default.yaml --student_name alice
```

This will:

* Train your model for the specified number of epochs.
* Save checkpoints in `checkpoints/alice/`.

You can edit:

* **Learning rate, EMA decay, epochs, and model width/depth** in your config file.
* Or **modify `models/model.py`** to build a new architecture (UNet, Transformer, etc.).

### 3. Generate Samples

After training, generate 5,000 samples for FID evaluation:

```bash
python sample.py --checkpoint checkpoints/alice/final.pt --student_name alice
```

Samples will be stored in:

```
outputs/alice/samples/
```

### 4. Evaluate Your Model

Run FID evaluation:

```bash
python evaluate.py --student_name alice
```

Results (FID score) will be saved in:

```
results/alice.json
```

### 5. Leaderboard

The instructor will run:

```bash
python leaderboard.py
```

This aggregates all results in `results/` and ranks students by FID.

---

## Rules

1. You **must** generate exactly **5,000 samples** for evaluation.
2. Use the **provided MNIST test set** for FID comparison.
3. Do not modify the evaluation scripts (only `model.py` and `configs` are allowed).
4. Submit:

   * Your `configs/my_config.yaml`
   * Your modified `models/model.py`
   * Your final checkpoint (`checkpoints/student_name/final.pt`)
   * (Optional) Your generated samples (`outputs/student_name/samples/`)

---

## Grading

* **Leaderboard FID score (60%)**: Lower is better.
* **Code quality & reproducibility (20%)**.
* **Report (20%)**: Short writeup (1 page) describing what you tried (hyperparameters, architecture changes, tricks).

Bonus points for:

* Novel architectures (e.g., Transformer backbone).
* Additional metrics (e.g., Precision/Recall).

---

## Tips

* Start simple: tune **learning rate, EMA decay, and UNet width** before modifying architecture.
* Use `configs/sample_transformer.yaml` if you want to try a Transformer backbone.
* For fast experiments, reduce epochs or use fewer samples during development (but final submission must use 5,000 samples).
