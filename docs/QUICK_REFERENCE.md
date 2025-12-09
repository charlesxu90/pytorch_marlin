# MARLIN PyTorch - Quick Reference Card

## ⚡ Data Format Performance

| Format | Load Time | File Size | Recommended |
|--------|-----------|-----------|-------------|
| HDF5 | 20-40 sec | 5.2 GB | ✅ **BEST** |
| NPZ | 40-60 sec | 6.8 GB | ✅ Good |
| CSV | 15-30 min | 12.5 GB | ❌ Slow |

**Always use HDF5 for MARLIN's 357,340 features!**

## Quick Commands

### Convert RData to HDF5 (Recommended)
```bash
Rscript convert_to_hdf5.R betas.RData y.RData training_data.h5
```

### Convert CSV to HDF5
```bash
python convert_csv_to_hdf5.py training_data.csv training_data.h5
```

### Extract Features
```bash
Rscript -e "load('marlin_v1.features.RData'); writeLines(betas_sub_names, 'reference_features.txt')"
```

### Export Labels
```bash
Rscript convert_rdata_to_csv.R y_to_csv y.RData labels.csv betas.RData
```

### Train Model
```bash
# With HDF5 (fast!)
python train.py --train_csv training_data.h5 --output_dir ./output --device cuda

# With CSV (slow)
python train.py --train_csv training_data.csv --output_dir ./output --device cuda
```

### Make Predictions
```bash
python predict.py \
    --model_path ./output/best_model.pt \
    --input_csv test_data.csv \
    --output_csv predictions.csv
```

### Monitor Training
```bash
tensorboard --logdir ./output/logs
```

## File Locations

| File | Purpose | Required |
|------|---------|----------|
| betas.RData | Methylation data | ✅ Yes |
| y.RData | Class labels | ✅ Yes |
| marlin_v1.features.RData | CpG names | ⚠️ Optional |
| training_data.h5 | Converted data | ✅ Created |
| reference_features.txt | Feature names | ⚠️ Optional |
| labels.csv | Label reference | ⚠️ Optional |

## Key Scripts

| Script | Purpose |
|--------|---------|
| convert_to_hdf5.R | RData → HDF5 (fastest) |
| convert_csv_to_hdf5.py | CSV → HDF5 |
| convert_rdata_to_csv.R | RData → CSV + labels |
| train.py | Train model |
| predict.py | Make predictions |
| model.py | Model architecture |
| data_utils_fast.py | Fast data I/O |

## Common Workflows

### Full Workflow (RData → Training)
```bash
# 1. Convert to HDF5
Rscript convert_to_hdf5.R betas.RData y.RData training_data.h5

# 2. Train
python train.py --train_csv training_data.h5 --output_dir ./output --device cuda

# 3. Predict
python predict.py --model_path ./output/best_model.pt --input_csv test.csv --output_csv pred.csv
```

### If You Have CSV Already
```bash
# 1. Convert CSV to HDF5 (one-time)
python convert_csv_to_hdf5.py training_data.csv training_data.h5

# 2. Train (same as above)
python train.py --train_csv training_data.h5 --output_dir ./output --device cuda
```

## Installation

```bash
# Basic
pip install -r requirements.txt

# GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For HDF5 support
pip install h5py
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train_csv` | - | Training data (.h5, .npz, or .csv) |
| `--output_dir` | ./output | Output directory |
| `--epochs` | 3000 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--samples_per_class` | 50 | Samples per class (upsampling) |
| `--device` | cuda | Device (cuda/cpu) |
| `--early_stopping_patience` | 100 | Early stopping patience |

## Model Architecture

```
Input:     357,340 CpG sites (binarized ±1)
Dropout:   99% (extreme regularization)
Hidden 1:  256 nodes (sigmoid)
Hidden 2:  128 nodes (sigmoid)
Output:    42 classes (softmax)

Total params: ~91.5 million
Model size: ~350 MB (PyTorch .pt)
```

## Output Files

After training:
```
output/
├── best_model.pt              # Best model (use this!)
├── final_model.pt             # Last epoch model
├── class_mapping.csv          # Class ID → name mapping
├── config.json                # Training config
├── training_history.json      # Loss/accuracy history
└── logs/                      # TensorBoard logs
```

## Data Formats

### Training CSV Format
```csv
label,cg00000029,cg00000165,...
AML_t_PML-RARA,0.123,0.456,...
ALL_B_PBXI,0.234,0.567,...
```

### Labels CSV Format
```csv
sample_id,original_label,merged_label,class_id,pb_merged
sample_1,AML_t_PML-RARA,AML_t_PML-RARA,0,FALSE
sample_2,PB_1,PB controls,3,TRUE
```

### Predictions CSV Format
```csv
sample_id,predicted_class,predicted_class_name,confidence,prob_class_0,...
sample_1,0,AML_t_PML-RARA,0.9876,0.9876,...
```

## Performance Tips

1. ✅ **Use HDF5 format** - 10-100x faster than CSV
2. ✅ **Use GPU** - 10-20x faster than CPU
3. ✅ **Increase batch size** - If you have GPU memory (32→64→128)
4. ✅ **Use early stopping** - Saves time, prevents overfitting
5. ✅ **Monitor with TensorBoard** - Track progress in real-time

## Troubleshooting

### Slow loading
→ Convert to HDF5 format

### CUDA out of memory
→ Reduce batch size: `--batch_size 16`

### Validation accuracy not improving
→ Check if data is binarized correctly
→ Try different learning rates

### Missing features
→ They'll be filled with 0 (unmethylated)
→ Warning will be shown

## Documentation

| File | Purpose |
|------|---------|
| README.md | Comprehensive documentation |
| QUICK_START.md | Step-by-step guide |
| FAST_DATA_LOADING.md | Performance optimization |
| OPTIMIZATION_SUMMARY.md | Detailed benchmarks |
| DATA_CONVERSION_GUIDE.md | Data conversion details |
| EXAMPLE_LABEL_CONVERSION.md | Label handling examples |

## Getting Help

1. Check documentation in `*.md` files
2. Run with `--help`: `python train.py --help`
3. Check TensorBoard logs for training issues
4. Verify data format with `head` command
5. Test with small dataset first

## Key Metrics

### MARLIN Dataset
- Samples: 2,356
- Features: 357,340 CpG sites
- Classes: 42 leukemia subtypes (after PB merging)

### File Sizes
- CSV: ~12.5 GB
- HDF5: ~5.2 GB (58% smaller)
- Model: ~350 MB

### Training Time (GPU)
- Per epoch: ~2-5 minutes
- Full training (3000 epochs): ~2-4 hours
- With early stopping: ~1-2 hours

## Quick Validation

### Check HDF5 file
```python
from data_utils_fast import load_training_data
data, labels, features = load_training_data('training_data.h5')
print(f"Data: {data.shape}, Labels: {len(set(labels))}, Features: {len(features)}")
```

### Check model
```python
from model import MARLINModel
model = MARLINModel.load_model('output/best_model.pt')
print(f"Parameters: {model.get_num_parameters():,}")
```

### Check predictions
```bash
head predictions.csv
# Should show: sample_id, predicted_class, predicted_class_name, confidence, ...
```

## Version Info

- Python: 3.8+
- PyTorch: 2.0+
- NumPy: 1.24+
- Pandas: 2.0+
- h5py: 3.8+

## Time Savings Summary

| Operation | CSV Time | HDF5 Time | Savings |
|-----------|----------|-----------|---------|
| First load | 18 min | 28 sec | 17 min |
| Per epoch | 1 min | 2 sec | 58 sec |
| 3000 epochs | 50 hours | 1.7 hours | 48 hours |

**Total time saved: ~48 hours per full training run!**
