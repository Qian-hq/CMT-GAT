# CMT-GAT: Contrastive Learning for Lane-Change Prediction under Missing Data

**CMT-GAT: A Contrastive Learning Framework for Predicting Lane Change Manoeuvres of Surrounding Vehicles in Weaving Areas under Missing Data Conditions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## Overview

CMT-GAT is a deep learning framework designed to predict lane-change intentions of surrounding vehicles in highway weaving areas, even when significant portions of sensor data are missing. It combines three key innovations:

1. **Temporal Trend-Aware Multi-Head Attention (TTA-MHA)** — captures temporal dynamics through causal convolution-based attention mechanisms
2. **Self-Supervised Contrastive Learning** — aligns representations of masked and original trajectories to ensure robustness under missing data
3. **Graph Attention Network (GAT)** — models spatial interactions among vehicles and environmental context for accurate intention classification

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py
```

## Repository Structure

```
CMT-GAT/
├── main.py                           # Main training script
├── models/
│   ├── __init__.py                   # Package init
│   ├── Contrastive_trend_at.py       # CMT-GAT model architecture
│   └── loss.py                       # Loss & evaluation functions
├── data/                             # Preprocessed dataset (not included)
├── output/                           # Training logs & model checkpoints
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
