# Fast CSV to HDF5 Conversion Guide

## The Problem

Loading very large CSV files with pandas is **extremely slow**:

- **MARLIN CSV**: 12.5 GB, 2,356 × 357,340
- **Pandas read_csv**: 15-30 minutes (or runs out of memory)
- **Issue**: Pandas has overhead for type inference, memory management, etc.

## The Solution

Use `csv_to_h5_fast.py` - a custom converter that **avoids pandas**:

- Uses Python's built-in `csv` module (C-optimized)
- Processes data in chunks (low memory usage)
- Writes directly to HDF5
- **Conversion time**: 2-5 minutes (one-time)
- **Loading time after**: 20-40 seconds (10-100x faster!)

## Quick Usage

### Basic Conversion

```bash
python csv_to_h5_fast.py training_data.csv training_data.h5
```

This will:
1. Read CSV in chunks (default 1000 rows)
2. Convert to numpy arrays incrementally
3. Write to HDF5 with compression
4. Show progress bar

### With Options

```bash
# Larger chunks (faster, more memory)
python csv_to_h5_fast.py training_data.csv training_data.h5 --chunk_size 5000

# No compression (faster conversion, larger file)
python csv_to_h5_fast.py training_data.csv training_data.h5 --compression none

# Verify after conversion
python csv_to_h5_fast.py training_data.csv training_data.h5 --verify

# Custom label column name
python csv_to_h5_fast.py data.csv output.h5 --label_column class_label
```

## Performance Comparison

### Conversion Methods

| Method | Time | Memory | Notes |
|--------|------|--------|-------|
| pandas read_csv + to_hdf5 | 15-30 min | 25-40 GB | Very slow |
| csv_to_h5_fast.py | 2-5 min | 2-4 GB | **Recommended** |

### After Conversion

| Format | Loading Time | File Size |
|--------|--------------|-----------|
| CSV | 15-30 min | 12.5 GB |
| HDF5 | 20-40 sec | 5.2 GB |

**Total time saved**: ~13-28 minutes per data load!

## How It Works

1. **First Pass**: Reads header, counts lines, determines dimensions
2. **Create HDF5**: Pre-allocates datasets with correct size
3. **Second Pass**: Reads CSV in chunks using built-in `csv.reader`
4. **Convert**: Converts each chunk to numpy array
5. **Write**: Writes chunks directly to HDF5
6. **Progress**: Shows progress bar with tqdm

## Key Advantages

### vs Pandas

- **No type inference overhead**: Direct string to float conversion
- **Chunked processing**: Doesn't load entire CSV into memory
- **Direct write**: No intermediate DataFrame structures
- **Built-in csv module**: C-optimized, very fast

### vs Other Methods

- **Memory efficient**: Only loads `chunk_size` rows at a time
- **Progress tracking**: Shows real-time progress
- **Error handling**: Handles missing/invalid values gracefully
- **Compression**: GZIP compression reduces file size by 40-60%

## Complete Example

### Starting with Large CSV

```bash
# 1. Install dependencies (if needed)
pip install h5py tqdm

# 2. Convert CSV to HDF5 (2-5 minutes)
python csv_to_h5_fast.py training_data.csv training_data.h5

# Expected output:
# Converting CSV to HDF5 (fast method)
#   Input: training_data.csv
#   Output: training_data.h5
#   Compression: gzip
#   Chunk size: 1000 rows
#   Input file size: 12.50 GB
#
# Counting lines in training_data.csv...
#   Total lines: 2,357 (including header)
#
# Reading header...
#   Samples: 2,356
#   Features: 357,340
#   Has labels: True
#
# Creating HDF5 file...
#
# Reading and converting data...
# Converting: 100%|████████████| 2356/2356 [02:34<00:00, 15.2rows/s]
#
# ✓ Conversion complete!
#   Input (CSV): 12.50 GB
#   Output (HDF5): 5.20 GB
#   Size reduction: 58.4%
#   Output file: training_data.h5

# 3. Train with fast loading!
python train.py --train_csv training_data.h5 --output_dir ./output
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--label_column` | `label` | Name of label column in CSV |
| `--compression` | `gzip` | Compression: gzip, lzf, or none |
| `--chunk_size` | `1000` | Rows to process at once |
| `--verify` | `False` | Verify HDF5 after conversion |

## Memory Usage

The script uses approximately:

```
Memory = chunk_size × n_features × 4 bytes + overhead
```

For MARLIN data (357,340 features):
- Chunk size 1000: ~1.4 GB
- Chunk size 5000: ~7 GB
- Chunk size 10000: ~14 GB

**Recommendation**: Use default chunk_size=1000 unless you have >16GB RAM

## Troubleshooting

### Out of Memory

Reduce chunk size:
```bash
python csv_to_h5_fast.py data.csv output.h5 --chunk_size 500
```

### Slow Conversion

Increase chunk size (if you have enough memory):
```bash
python csv_to_h5_fast.py data.csv output.h5 --chunk_size 5000
```

### No Progress Bar

Install tqdm:
```bash
pip install tqdm
```

### Invalid Values in CSV

The script automatically:
- Replaces empty/missing values with 0.0
- Warns about invalid values
- Continues processing

## Verification

Verify the HDF5 file:

```bash
python csv_to_h5_fast.py data.csv output.h5 --verify
```

Or manually:

```python
import h5py
import numpy as np

with h5py.File('training_data.h5', 'r') as f:
    print("Datasets:", list(f.keys()))
    print("Data shape:", f['data'].shape)
    print("Labels shape:", f['labels'].shape)
    print("Features:", len(f['feature_names']))

    # Load first 10 samples to verify
    data_sample = f['data'][:10]
    labels_sample = f['labels'][:10]
    print("\nFirst 10 labels:", labels_sample)
```

## CSV Format Requirements

The CSV must have:
1. **Header row**: Column names
2. **Label column**: Named 'label' (or specify with `--label_column`)
3. **Feature columns**: Numeric values (0-1 for beta values)
4. **Format**: Standard CSV (comma-separated)

Example:
```csv
label,cg00000029,cg00000165,cg00000236,...
AML_t_PML-RARA,0.123,0.456,0.789,...
AML_t_RUNX1-RUNX1T1,0.234,0.567,0.890,...
```

## When to Use This

**Use csv_to_h5_fast.py when:**
- ✅ You have a very large CSV (>1 GB)
- ✅ Pandas read_csv is too slow
- ✅ You want to train multiple times (one-time conversion)
- ✅ You need memory-efficient processing

**Don't need it if:**
- ❌ Your CSV is small (<100 MB)
- ❌ You only need to load data once
- ❌ You already have HDF5 format

## Complete Workflow

### From RData (Recommended)

```bash
# Convert RData directly to HDF5 (fastest)
python convert_to_h5.py \
    --betas betas.RData \
    --labels y.RData \
    --output training_data.h5 \
    --format h5
```

### From Large CSV (If You Already Have It)

```bash
# Convert CSV to HDF5 (one-time, 2-5 minutes)
python csv_to_h5_fast.py training_data.csv training_data.h5

# Then train (fast loading, 20-40 seconds)
python train.py --train_csv training_data.h5 --output_dir ./output
```

## Comparison with Other Scripts

| Script | Input | Output | Speed | Use Case |
|--------|-------|--------|-------|----------|
| convert_to_h5.py | RData | H5/CSV | Fast | Direct RData conversion |
| csv_to_h5_fast.py | CSV | H5 | Medium | Large CSV conversion |
| convert_rdata_to_csv.R | RData | CSV | Slow | If you need CSV |

**Recommendation**:
- Have RData? Use `convert_to_h5.py --format h5`
- Have large CSV? Use `csv_to_h5_fast.py`
- Need CSV? Use `convert_to_h5.py --format csv`

## Summary

**Problem**: Loading large CSV is extremely slow (15-30 min)
**Solution**: Convert CSV to HDF5 once (2-5 min)
**Result**: Load in 20-40 seconds (10-100x faster!)

**Total time saved**: ~13-28 minutes per training run!

```bash
# Simple command to remember:
python csv_to_h5_fast.py training_data.csv training_data.h5
```
