# MARLIN PyTorch - Quick Start Guide

## Installation (5 minutes)

```bash
cd pytorch_marlin
pip install -r requirements.txt
```

For GPU support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## ⚡ Important: Data Format

**CSV is VERY SLOW for MARLIN data!**

- CSV loading: 15-30 minutes
- HDF5 loading: 20-40 seconds (**15-60x faster!**)

**✅ Recommended:** Use HDF5 format (see below)

See [FAST_DATA_LOADING.md](FAST_DATA_LOADING.md) for details.

## Step 1: Prepare Your Data

### Option A: Convert to HDF5 Format (RECOMMENDED - FAST!)

If you have the original MARLIN `betas.RData` and `y.RData` files:

```bash
# Convert directly to HDF5 (10-100x faster than CSV!)
Rscript convert_to_hdf5.R \
      ../MARLIN/betas.RData \
      ../MARLIN/y.RData \
      training_data.h5

# This takes 2-5 minutes but saves hours of loading time!
```

### Option B: Already Have Large CSV? Convert to HDF5!

If you already have a CSV file and loading it is very slow:

```bash
# Convert CSV to HDF5 (2-5 minutes, one-time)
python csv_to_h5_fast.py training_data.csv training_data.h5

# This script:
# - Avoids pandas (uses built-in csv module - much faster!)
# - Processes in chunks (low memory)
# - Shows progress bar
# - 10-100x faster loading after conversion!

# See CSV_TO_H5_GUIDE.md for details
```

### Option C: Convert to CSV (Slower but Compatible)

If you prefer CSV format:

```bash
# Convert MARLIN training data (beta values + labels)
python convert_to_h5.py \
      --betas ../MARLIN/betas.RData \
      --labels ../MARLIN/y.RData \
      --output training_data.csv \
      --format csv

# Obtain the sample labels
Rscript convert_rdata_to_csv.R y_to_csv \
      ../MARLIN/y.RData \
      labels.csv \
      ../MARLIN/betas.RData

# Extract reference features (357,340 CpG sites) - optional
Rscript convert_rdata_to_csv.R extract_features \
    ../MARLIN/MARLIN_realtime/files/marlin_v1.features.RData \
    reference_features.txt
```

### Option D: Convert BED Files

If you have multiple BED files from nanopore sequencing:

```bash
# Create sample labels CSV first (sample_id,label)
# Example: sample_001,AML_t_PML-RARA

# Convert BED files to training matrix
python convert_data.py \
    --mode beds_to_matrix \
    --input "path/to/bed/files/*.bed" \
    --output training_data.csv \
    --labels sample_labels.csv \
    --reference_features reference_features.txt
```

### Option C: Prepare Your Own CSV

Create a CSV file with this format:

```csv
label,cg00000029,cg00000165,cg00000236,...
AML_t_PML-RARA,0.123,0.456,0.789,...
AML_t_RUNX1-RUNX1T1,0.234,0.567,0.890,...
```

## Step 2: Train the Model

### Quick Training (for testing)

```bash
# With HDF5 (fast loading!)
python train.py \
    --train_csv training_data.h5 \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 32 \
    --device cuda

# Or with CSV (slow loading)
python train.py \
    --train_csv training_data.csv \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 32 \
    --device cuda
```

### Full Training (production)

```bash
# With HDF5 (recommended for large datasets)
python train.py \
    --train_csv training_data.h5 \
    --output_dir ./output \
    --reference_features reference_features.txt \
    --epochs 3000 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --samples_per_class 50 \
    --device cuda
```

**Note:** The script auto-detects format (.h5, .npz, or .csv) by extension!

Training time:
- GPU: ~2-4 hours
- CPU: ~24+ hours

Monitor with TensorBoard:
```bash
tensorboard --logdir ./output/logs
```

## Step 3: Make Predictions

```bash
python predict.py \
    --model_path ./output/best_model.pt \
    --input_csv test_samples.csv \
    --output_csv predictions.csv \
    --class_mapping ./output/class_mapping.csv \
    --device cuda
```

## Step 4: View Results

```bash
# View predictions
head predictions.csv

# Or in Python
python -c "
import pandas as pd
df = pd.read_csv('predictions.csv')
print(df[['sample_id', 'predicted_class_name', 'confidence']].head())
"
```

## Example Workflow

Run the complete example:

```bash
python example_workflow.py
```

This will:
1. Create dummy training data
2. Train a model
3. Make predictions
4. Show results

## Common Use Cases

### Case 1: I have the original MARLIN data

```bash
# 1. Convert data (betas.RData + y.RData)
Rscript convert_rdata_to_csv.R marlin_to_csv \
    betas.RData \
    y.RData \
    training_data.csv

# Optional: Extract reference features
Rscript convert_rdata_to_csv.R extract_features \
    marlin_v1.features.RData \
    reference_features.txt

# 2. Train
python train.py \
    --train_csv training_data.csv \
    --output_dir ./output \
    --device cuda

# 3. Predict
python predict.py \
    --model_path ./output/best_model.pt \
    --input_csv new_samples.csv \
    --output_csv predictions.csv
```

### Case 2: I have BED files from nanopore sequencing

```bash
# 1. Extract features from reference
python convert_data.py --mode extract_features --input marlin_v1.probes_hg38.bed.gz --output reference_features.txt

# 2. Convert BED files to matrix
python convert_data.py --mode beds_to_matrix --input "data/*.bed" --output training_data.csv --labels labels.csv

# 3. Train and predict (same as above)
```

### Case 3: I want to use the model in my Python code

```python
from model import MARLINModel
import torch
import pandas as pd
import numpy as np

# Load model
model = MARLINModel.load_model('output/best_model.pt', device='cuda')

# Prepare data (binarized methylation values)
data = pd.read_csv('sample.csv')
features = data.drop(columns=['label']).values
features = np.where(features >= 0.5, 1, -1).astype(np.float32)

# Predict
with torch.no_grad():
    input_tensor = torch.FloatTensor(features).cuda()
    probabilities = model.predict_proba(input_tensor)
    predictions = probabilities.argmax(dim=1)

print(f"Predictions: {predictions.cpu().numpy()}")
```

## Troubleshooting

**CUDA out of memory?**
```bash
python train.py --batch_size 16 ...  # Reduce batch size
```

**Training too slow?**
```bash
# Use GPU
python train.py --device cuda ...

# Or reduce data/epochs for testing
python train.py --epochs 100 --samples_per_class 20 ...
```

**Missing features?**
- The script will warn but continue with available features
- Missing features are filled with 0 (unmethylated)

## File Structure

After training, you'll have:

```
output/
├── best_model.pt              # Use this for predictions
├── class_mapping.csv          # Class index to name mapping
├── config.json                # Training configuration
├── training_history.json      # Loss/accuracy per epoch
└── logs/                      # TensorBoard logs
```

## Next Steps

1. **Evaluate model**: Check `training_history.json` for accuracy
2. **Fine-tune**: Adjust hyperparameters if needed
3. **Deploy**: Use `best_model.pt` for production predictions
4. **Monitor**: Use TensorBoard to visualize training

## Getting Help

- **Documentation**: See `README.md` for detailed information
- **Examples**: Run `python example_workflow.py`
- **Code**: All scripts have `--help` option

```bash
python train.py --help
python predict.py --help
python convert_data.py --help
```
