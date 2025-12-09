# MARLIN Data Conversion Guide

## Understanding MARLIN Training Data

The original MARLIN training requires **TWO** critical data files:

### 1. betas.RData
- Contains methylation beta values matrix
- Dimensions: 2,356 samples × 357,340 CpG sites
- Values: continuous beta values (0 to 1)

### 2. y.RData (CRITICAL - DO NOT SKIP)
- Contains class labels for each sample
- 42 unique leukemia subtypes
- Example labels: `AML_t_PML-RARA`, `ALL_B_PBXI`, `PB_1`, `PB_2`, etc.

**Without y.RData, you cannot train the model because you need labels!**

## Label Preprocessing

The original MARLIN applies important preprocessing to labels:

```r
# Merge all peripheral blood controls into one class
y <- as.character(y)
y <- ifelse(grepl("^PB", y) == TRUE, "PB controls", y)
y <- as.factor(y)
```

This converts:
- `PB_1`, `PB_2`, `PB_3`, ... → `PB controls` (single class)
- All other labels remain unchanged
- Result: typically ~42 classes after merging

## Correct Conversion Method

### Method 1: Convert Full Training Data (Betas + Labels)

```bash
# CORRECT: Convert both betas.RData and y.RData together
Rscript convert_rdata_to_csv.R marlin_to_csv \
    betas.RData \
    y.RData \
    training_data.csv
```

This will:
1. Load beta values from betas.RData
2. Load class labels from y.RData
3. Merge PB controls automatically
4. Create CSV with format: `label,cg00000029,cg00000165,...`
5. Show class distribution before and after merging

### Method 2: Convert Labels Only (NEW)

If you want to inspect or export just the labels:

```bash
# Convert y.RData to labels CSV
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv

# Optional: Include sample IDs from betas.RData
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv \
    betas.RData
```

This creates a CSV with columns:
- `sample_id` - Sample identifier
- `original_label` - Original class label from y.RData
- `merged_label` - Label after merging PB controls
- `class_id` - Numeric class ID (0-based)
- `pb_merged` - Boolean flag indicating if PB control was merged

### Output Format

**Full Training Data (marlin_to_csv):**

```csv
label,cg00000029,cg00000165,cg00000236,...
AML_t_PML-RARA,0.123,0.456,0.789,...
AML_t_RUNX1-RUNX1T1,0.234,0.567,0.890,...
ALL_B_PBXI,0.345,0.678,0.901,...
PB controls,0.456,0.789,0.012,...
...
```

**Labels Only (y_to_csv):**

```csv
sample_id,original_label,merged_label,class_id,pb_merged
sample_1,AML_t_PML-RARA,AML_t_PML-RARA,0,FALSE
sample_2,AML_t_RUNX1-RUNX1T1,AML_t_RUNX1-RUNX1T1,1,FALSE
sample_3,ALL_B_PBXI,ALL_B_PBXI,2,FALSE
sample_4,PB_1,PB controls,3,TRUE
sample_5,PB_2,PB controls,3,TRUE
...
```

## Common Mistakes

### ❌ WRONG: Converting only betas.RData

```bash
# This creates data WITHOUT labels - cannot be used for training
Rscript convert_rdata_to_csv.R betas_to_csv betas.RData data.csv
```

### ✅ CORRECT: Converting both files

```bash
# This creates data WITH labels - ready for training
Rscript convert_rdata_to_csv.R marlin_to_csv betas.RData y.RData training_data.csv
```

## Verification

After conversion, verify your data:

```bash
# Check the CSV structure
head -n 1 training_data.csv  # Should show: label,cg00000029,...

# Count samples
wc -l training_data.csv  # Should show 2,357 lines (2,356 samples + 1 header)

# Check label column
cut -d',' -f1 training_data.csv | sort | uniq -c
# Should show distribution of classes
```

Or in Python:

```python
import pandas as pd

df = pd.read_csv('training_data.csv')
print(f"Shape: {df.shape}")  # Should be (2356, 357341)
print(f"\nFirst column: {df.columns[0]}")  # Should be 'label'
print(f"\nClass distribution:")
print(df['label'].value_counts())
```

## Complete Workflow

### Step 1: Locate Original Data

Find these files from the original MARLIN repository:
- `betas.RData` - methylation data
- `y.RData` - class labels
- `marlin_v1.features.RData` - feature names (optional)

### Step 2: Convert to CSV

```bash
# Convert training data
Rscript convert_rdata_to_csv.R marlin_to_csv \
    /path/to/betas.RData \
    /path/to/y.RData \
    training_data.csv

# Optional: Extract feature names
Rscript convert_rdata_to_csv.R extract_features \
    /path/to/marlin_v1.features.RData \
    reference_features.txt
```

### Step 3: Verify Conversion

```bash
# Quick check
head -n 2 training_data.csv
```

Expected output:
```
label,cg00000029,cg00000165,...
AML_t_PML-RARA,0.123,0.456,...
```

### Step 4: Train Model

```bash
python train.py \
    --train_csv training_data.csv \
    --output_dir ./output \
    --epochs 3000 \
    --batch_size 32 \
    --device cuda
```

## Data Files Summary

| File | Required? | Contains | Size |
|------|-----------|----------|------|
| betas.RData | ✅ Required | Methylation beta values | ~200 MB |
| y.RData | ✅ Required | Class labels | ~100 KB |
| marlin_v1.features.RData | ⚠️ Optional | CpG probe names | ~1.7 MB |
| marlin_v1.probes_*.bed.gz | ⚠️ Optional | Genomic coordinates | ~5-6 MB |

## Troubleshooting

### "Could not find 'y' object in y.RData"

Make sure you're loading the correct y.RData file:

```bash
# Check what's inside the RData file
Rscript -e "load('y.RData'); print(ls()); print(length(y)); print(table(y))"
```

Should show:
- Object named 'y' exists
- Length ~2,356 (matching number of samples)
- Table showing distribution across 42+ classes

### "Length of class_labels does not match"

This means the number of labels doesn't match the number of samples:

```bash
# Check dimensions
Rscript -e "load('betas.RData'); load('y.RData'); cat('Betas:', nrow(betas), 'x', ncol(betas), '\n'); cat('Labels:', length(y), '\n')"
```

Should show:
```
Betas: 2356 x 357340
Labels: 2356
```

### Missing PB Controls Merge

If you see separate `PB_1`, `PB_2`, etc. classes in your data:

```python
import pandas as pd

df = pd.read_csv('training_data.csv')

# Manually merge PB controls
df['label'] = df['label'].apply(lambda x: 'PB controls' if str(x).startswith('PB') else x)

df.to_csv('training_data_merged.csv', index=False)
```

## Additional Data Augmentation (Optional)

The original MARLIN training also applies 10% CpG flipping after upsampling:

```r
# Flip 10% of CpGs randomly (data augmentation)
flip_x_percent <- function(x, x_percent = 0.1) {
  sample_indices <- sample(1:length(x), x_percent * length(x))
  x[sample_indices] <- -x[sample_indices]
  return(x)
}
```

This is **optional** in PyTorch as the 99% dropout provides strong regularization. If you want to add this, modify the `MARLINDataset` class in `data_utils.py`.

## Questions?

- **Q: Can I train without y.RData?**
  - A: No, you need labels for supervised learning.

- **Q: What if I only have BED files?**
  - A: See the BED conversion section in README.md. You'll need to create your own labels CSV.

- **Q: Why merge PB controls?**
  - A: It's part of the original MARLIN preprocessing to combine all peripheral blood controls into one reference class.

- **Q: Can I skip the merging?**
  - A: Yes, but it won't match the original MARLIN training. To skip: manually edit the R script.
