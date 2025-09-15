# Codespaces Template — Software Automation (Lesson 7)

This repository is a **fully preloaded GitHub Codespaces template** for Lesson 7.
It bakes Python dependencies into the container image for **instant startup** and
saves plots headlessly to `out/` (no GUI required).

## Quick Start
1. Push this repository to GitHub.
2. Open **Code → Create codespace on main**.
3. In the terminal, run:
   ```bash
   python train.py
   ```
   Optional flags:
   ```bash
   python train.py --poly_degree 4 --logit_threshold 0.6
   python train.py --csv data/sample_marks.csv
   ```

## What's Inside
- `.devcontainer/Dockerfile` — builds an image with **preinstalled** requirements.
- `.devcontainer/devcontainer.json` — points Codespaces to the Dockerfile.
- `requirements.txt` — pinned versions for reproducibility.
- `src/` — OOP models, metrics, plotting (Agg backend), data loading.
- `data/sample_marks.csv` — small dataset.
- `train.py` — runs linear, polynomial, and logistic models; writes PNGs to `out/`.

## Enable Prebuilds (faster for classes)
To make codespaces **warm and ready** for students:
1. Push this repo to GitHub (public or private).
2. Go to **Settings → Codespaces → Prebuilds** and enable prebuilds for the default
   branch (e.g., `main`). Choose the region closest to your class.
3. (Optional) Use the included workflow below to build the devcontainer image on every push.
