# MARLIN PyTorch Implementation

PyTorch implementation of **MARLIN** (Methylation- and AI-guided Rapid Leukemia Subtype Inference) for classifying acute leukemia subtypes using DNA methylation profiles.

## Overview

This is a reimplementation of MARLIN using PyTorch instead of TensorFlow/Keras, with CSV-based data format for easier integration with standard machine learning workflows.


## Installation

### Requirements

```bash
pip install torch numpy pandas scikit-learn tensorboard
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

This creates a reference CSV with columns: `sample_id`, `original_label`, `merged_label`, `class_id`, `pb_merged`

#### Option B: Convert BED Files

If you have BED files from nanopore sequencing:

```bash
# Extract reference features from reference BED file
python convert_data.py \
    --mode extract_features \
    --input /path/to/marlin_v1.probes_hg38.bed.gz \
    --output reference_features.txt

# Convert multiple BED files to training matrix
python convert_data.py \
    --mode beds_to_matrix \
    --input "data/*.bed" \
    --output training_data.csv \
    --labels sample_labels.csv \
    --reference_features reference_features.txt
```

**Note:** `sample_labels.csv` should have columns: `sample_id,label`

### 2. Train the Model

```bash
python train.py \
    --train_csv training_data.csv \
    --output_dir ./output \
    --reference_features reference_features.txt \
    --epochs 3000 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --device cuda
```

**Training Options:**
- `--train_csv`: Path to training data CSV
- `--output_dir`: Directory to save model and logs
- `--reference_features`: Path to reference features file (optional)
- `--epochs`: Number of training epochs (default: 3000)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--samples_per_class`: Samples per class after upsampling (default: 50)
- `--no_upsample`: Disable class upsampling
- `--early_stopping_patience`: Patience for early stopping (default: 100)
- `--device`: Device to train on (cuda/cpu)
- `--seed`: Random seed for reproducibility (default: 42)

### 3. Make Predictions

```bash
python predict.py \
    --model_path ./output/best_model.pt \
    --input_csv test_data.csv \
    --output_csv predictions.csv \
    --reference_features reference_features.txt \
    --class_mapping ./output/class_mapping.csv \
    --batch_size 32 \
    --device cuda
```

**Prediction Options:**
- `--model_path`: Path to trained model (.pt file)
- `--input_csv`: Path to input data CSV
- `--output_csv`: Path to save predictions
- `--reference_features`: Path to reference features file (optional)
- `--class_mapping`: Path to class mapping CSV (optional)
- `--batch_size`: Batch size for prediction (default: 32)
- `--no_probabilities`: Don't include full probability distribution in output
- `--device`: Device to use (cuda/cpu)

## Data Format

### Training Data CSV Format

The training CSV should have the following structure:

```csv
label,cg00000029,cg00000165,cg00000236,...
AML_t_PML-RARA,0.123,0.456,0.789,...
AML_t_RUNX1-RUNX1T1,0.234,0.567,0.890,...
ALL_B_PBXI,0.345,0.678,0.901,...
...
```

- **First column (`label`):** Class label for each sample
- **Remaining columns:** Methylation beta values (0-1) for each CpG site
- **Column names:** Should match CpG probe IDs

### Input Data for Prediction

For prediction, the CSV format is similar but the `label` column is optional:

```csv
sample_id,cg00000029,cg00000165,cg00000236,...
sample_001,0.123,0.456,0.789,...
sample_002,0.234,0.567,0.890,...
```

### Reference Features File

A simple text file with one feature (CpG probe ID) per line:

```
cg00000029
cg00000165
cg00000236
...
```

## Output Files

### After Training

```
output/
├── best_model.pt              # Best model based on validation accuracy
├── final_model.pt             # Model from last epoch
├── checkpoint_epoch_*.pt      # Periodic checkpoints
├── class_mapping.csv          # Mapping of class indices to names
├── config.json                # Training configuration and results
├── training_history.json      # Loss and accuracy per epoch
└── logs/                      # TensorBoard logs
```

### After Prediction

```
predictions.csv
```

Columns:
- `sample_id`: Sample identifier
- `predicted_class`: Predicted class index
- `predicted_class_name`: Predicted class name (if class mapping provided)
- `true_label`: True label (if available in input)
- `confidence`: Maximum probability (confidence score)
- `prob_<class_name>`: Probability for each class (if `--no_probabilities` not set)

## Model Usage in Python

### Training

```python
from train import train_marlin

model, history = train_marlin(
    train_csv='training_data.csv',
    output_dir='./output',
    epochs=3000,
    batch_size=32,
    learning_rate=1e-5,
    device='cuda'
)
```

### Prediction

```python
from predict import predict_batch

results = predict_batch(
    model_path='./output/best_model.pt',
    input_csv='test_data.csv',
    output_csv='predictions.csv',
    device='cuda'
)
```

### Single Sample Prediction

```python
from predict import predict_single

result = predict_single(
    model_path='./output/best_model.pt',
    input_csv='single_sample.csv',
    class_mapping='./output/class_mapping.csv',
    device='cuda'
)

print(f"Predicted class: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.4f}")
print("Top 5 predictions:")
for pred in result['top_predictions']:
    print(f"  {pred['class_name']}: {pred['probability']:.4f}")
```

### Loading and Using Model Directly

```python
import torch
from model import MARLINModel

# Load model
model = MARLINModel.load_model('best_model.pt', device='cuda')
model.eval()

# Prepare input data (binarized methylation values)
# Shape: (batch_size, 357340)
input_data = torch.FloatTensor(data).cuda()

# Get predictions
with torch.no_grad():
    probabilities = model.predict_proba(input_data)
    predictions = probabilities.argmax(dim=1)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./output/logs
```

Then open http://localhost:6006 in your browser.

## Performance Considerations

### GPU vs CPU

- **GPU (recommended):** Training takes ~2-4 hours on modern GPU
- **CPU:** Training can take 24+ hours

### Memory Requirements

- **GPU:** ~4-8 GB VRAM
- **CPU:** ~16-32 GB RAM
- **Storage:** ~2-5 GB for model and data

### Batch Size

- Larger batch sizes (64, 128) are faster but require more memory
- Smaller batch sizes (16, 32) are slower but more memory-efficient
- Default of 32 is a good balance

## Differences from Original MARLIN

| Feature | Original MARLIN | PyTorch MARLIN |
|---------|----------------|----------------|
| Framework | TensorFlow/Keras (R) | PyTorch (Python) |
| Data Format | BED files + RData | CSV files |
| Real-time | Yes (nanopore sequencing) | No |
| Web Interface | Shiny app | Not included |
| Batch Processing | R parallel | PyTorch DataLoader |
| Model Format | HDF5 | PyTorch .pt |
| Data Augmentation | 10% CpG flipping | Not implemented (optional) |

### Important Notes on Data Preprocessing

The original MARLIN training applies these preprocessing steps:

1. **Merge PB Controls**: All peripheral blood control samples (labels starting with "PB") are merged into a single "PB controls" class
2. **Binarization**: Beta values ≥0.5 → +1, <0.5 → -1
3. **Upsampling**: 50 random samples per class (with replacement)
4. **Data Augmentation**: 10% of CpG sites randomly flipped (optional noise injection)

**The PyTorch implementation handles steps 1-3.** Step 4 (CpG flipping) is not implemented as the 99% dropout already provides strong regularization. If you need exact reproduction, you can add noise augmentation to the dataset class.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 16 ...
```

### Missing Features

If reference features are missing from input data, the script will:
1. Warn you about missing features
2. Fill missing values with 0 (unmethylated)
3. Continue with available features

### Class Imbalance

Enable upsampling (default) to balance classes:
```bash
python train.py --samples_per_class 50 ...
```

To disable upsampling:
```bash
python train.py --no_upsample ...
```

## Citation

If you use this implementation, please cite the original MARLIN paper:

```
Hovestadt V, et al. (2024)
MARLIN: Methylation- and AI-guided Rapid Leukemia Subtype Inference
[Add publication details when available]
```

## License

This implementation follows the same license as the original MARLIN project.

## Contact

For questions about the PyTorch implementation, please open an issue in this repository.

For questions about the original MARLIN method, contact the Hovestadt Lab at Dana-Farber Cancer Institute.
