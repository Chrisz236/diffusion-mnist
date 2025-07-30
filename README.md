# Diffusion-MNIST Assignment

Train and tune a **diffusion model** on the MNIST dataset using a UNet backbone.  
This repo provides a modular baseline implementation. Your task is to explore architecture or training improvements to generate better samples, and compete for the **lowest FID**!

---

## Repository Structure

```

diffusion-mnist/
├── configs/           # Config files for training
│   └── default.yaml   # Baseline config
├── data/              # MNIST auto-downloads here
├── models/            # Model architecture
│   ├── model.py       # Unified interface (you can modify this)
│   └── unet.py        # Provided UNet baseline
├── outputs/           # Checkpoints and sample images
├── train.py           # Train a model
├── sample.py          # Generate image grid from a checkpoint
├── evaluate.py        # Compute FID vs. MNIST test set
├── submit.py          # Package submission (optional helper)
├── environment.yml    # Conda environment spec
└── README.md          # You are here

````

---

## Quick Start

### 1. Setup Conda environment

```bash
conda env create -f environment.yml
conda activate diffusion-mnist
````

### 2. Train the baseline model

```bash
python train.py --config configs/default.yaml
```

By default, checkpoints and sample images are saved under `outputs/`.

### 3. Sample from a trained model

```bash
python sample.py --ckpt outputs/ckpt_100.pt
```

This will generate an `8x8` grid of samples as a `.png` file.

### 4. Evaluate FID

```bash
python evaluate.py --ckpt outputs/ckpt_100.pt
```

This computes the FID between 10,000 generated samples and the MNIST test set.

---

## Configuration

All hyperparameters are set in `configs/default.yaml`, including:

* Training: `epochs`, `batch_size`, `lr`, `ema_decay`
* Diffusion: `T` (total time steps), `beta_schedule`
* Paths: `data_root`, `output_root`

You can modify this file or pass in a different config using:

```bash
python train.py --config configs/my_config.yaml
```

---

## Assignment Rules

You're expected to **modify `models/model.py`** as your main interface to define the model architecture and noise schedule. You may:

- ✅ Use new architectures (e.g., Transformer)
- ✅ Change how time embeddings are used
- ✅ Tweak the noise schedule or loss function
- ✅ Add new files under `models/`

But do **not** change:

* CLI and config interfaces of `train.py`, `sample.py`, or `evaluate.py`
* Output paths or file naming convention

---

## Submission Format

Your final submission should include:

```
yourname_submission.zip
├── models/
│   ├── model.py             # Your modified model interface
│   └── *.py                 # Any extra files (e.g. transformer.py)
├── ckpt_final.pt            # Final checkpoint
├── final_grid.png           # 8x8 sampled image grid from final model
└── results.txt              # One line: FID = <value>
```

We will unzip and run:

```bash
python sample.py   --ckpt ckpt_final.pt
python evaluate.py --ckpt ckpt_final.pt
```

To verify your claimed results.

Happy diffusing!