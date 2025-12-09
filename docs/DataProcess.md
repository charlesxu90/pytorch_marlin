# Pytorch MARLIN data processing

## Convert Rdata to csv
> **⚡ Performance Note:** HDF5 format is 10-100x faster than CSV for MARLIN's 357,340 features!
> See [FAST_DATA_LOADING.md](FAST_DATA_LOADING.md) and [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for details.

### 1. Convert Existing Data

#### Option A: Convert to HDF5 Format (FASTEST - RECOMMENDED)

```bash
# Direct RData to HDF5 conversion (10-100x faster loading!)
Rscript convert_to_hdf5.R \
    betas.RData \
    y.RData \
    training_data.h5
```

This takes 2-5 minutes but saves hours of loading time during training!

#### Option B: Convert Original MARLIN RData to CSV

If you have the original MARLIN training data (`betas.RData` and `y.RData`):

```bash
# Convert MARLIN training data to CSV (includes label preprocessing)
Rscript convert_rdata_to_csv.R marlin_to_csv \
    betas.RData \
    y.RData \
    training_data.csv
```

This will:
- Load both beta values and class labels
- Merge peripheral blood controls (PB* → "PB controls")
- Create a CSV with format: `label,cg00000029,cg00000165,...`

You can also export labels separately for inspection:

```bash
# Export y.RData to labels CSV (with original and merged labels)
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv \
    betas.RData
```

This creates a reference CSV with columns: `sample_id`, `original_label`, `merged_label`, `class_id`, `pb_merged`

#### Option C: Convert BED Files

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

## Convert CSV to H5

Loading very large CSV files with pandas is **extremely slow**:

- **MARLIN CSV**: 12.5 GB, 2,356 × 357,340
- **Pandas read_csv**: 15-30 minutes (or runs out of memory)
- **Issue**: Pandas has overhead for type inference, memory management, etc.


Use `csv_to_h5_fast.py` - a custom converter that **avoids pandas**:

- Uses Python's built-in `csv` module (C-optimized)
- Processes data in chunks (low memory usage)
- Writes directly to HDF5
- **Conversion time**: 2-5 minutes (one-time)
- **Loading time after**: 20-40 seconds (10-100x faster!)


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
# Simple command to use:
python csv_to_h5_fast.py training_data.csv training_data.h5

# Larger chunks (faster, more memory)
python csv_to_h5_fast.py training_data.csv training_data.h5 --chunk_size 5000

# No compression (faster conversion, larger file)
python csv_to_h5_fast.py training_data.csv training_data.h5 --compression none

# Verify after conversion
python csv_to_h5_fast.py training_data.csv training_data.h5 --verify

# Custom label column name
python csv_to_h5_fast.py data.csv output.h5 --label_column class_label
```


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