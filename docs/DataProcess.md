# Pytorch MARLIN data processing

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