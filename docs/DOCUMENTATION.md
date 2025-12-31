# POST-HCT Survival Prediction Project Documentation

Last updated: 2025-12-31

## Overview

This repository contains resources and a Jupyter notebook for (POST-HCT). The main work is in `post_hct.ipynb` using the provided CSV data files.

## Files

- `post_hct.ipynb` — Primary analysis notebook.
- `train.csv` — Main dataset used for training/analysis.
- `data_dictionary.csv` — Column descriptions and data dictionary.
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10+ (virtual environment recommended)
- Jupyter (or JupyterLab)

## Setup

1. Create and activate a virtual environment (example using venv):

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1    # on Windows PowerShell
pip install -r requirements.txt
```

2. Launch the notebook server:

```bash
jupyter notebook post_hct.ipynb
```

## Usage

Open `post_hct.ipynb` and run cells sequentially. The notebook reads `train.csv` and references `data_dictionary.csv` for column meanings.

## Contact

Repository maintainer: Arjun
