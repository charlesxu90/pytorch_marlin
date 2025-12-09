# MARLIN PyTorch model

**Key Features:**
- PyTorch-based neural network (357,340 → 256 → 128 → 42)
- ⚡ **Fast HDF5 format (10-100x faster than CSV!)**
- CSV data format support
- Batch training and prediction
- Early stopping and model checkpointing
- TensorBoard logging
- GPU acceleration support

## Architecture

```
Input Layer:     357,340 CpG sites (binarized to +1/-1)
                    ↓
Dropout Layer:   99% dropout (extreme regularization)
                    ↓
Hidden Layer 1:  256 nodes (sigmoid activation)
                    ↓
Hidden Layer 2:  128 nodes (sigmoid activation)
                    ↓
Output Layer:    42 leukemia subtypes (softmax)
```

**Total Parameters:** ~91.5 million


## Directory Structure

```
pytorch_marlin/
├── model.py              # MARLIN model architecture
├── data_utils.py         # Data loading and preprocessing utilities
├── train.py              # Training script
├── predict.py            # Prediction script
├── convert_data.py       # Data conversion utilities
└── README.md            # This file
```
