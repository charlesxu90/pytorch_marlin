# Example: Converting y.RData to Labels CSV

## Why Export Labels Separately?

Exporting labels to a separate CSV is useful for:

1. **Inspection**: Quickly view class distribution and sample assignments
2. **Validation**: Verify which samples are merged (PB controls)
3. **Reference**: Create a lookup table for sample IDs and class names
4. **Analysis**: Analyze label distribution without loading full methylation data

## Usage Examples

### Example 1: Basic Label Export

```bash
# Convert y.RData to labels CSV
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv
```

**Output (labels.csv):**
```csv
sample_id,original_label,merged_label,class_id,pb_merged
sample_1,AML_t_PML-RARA,AML_t_PML-RARA,0,FALSE
sample_2,AML_t_RUNX1-RUNX1T1,AML_t_RUNX1-RUNX1T1,1,FALSE
sample_3,ALL_B_PBXI,ALL_B_PBXI,2,FALSE
sample_4,PB_1,PB controls,3,TRUE
sample_5,PB_2,PB controls,3,TRUE
sample_6,PB_3,PB controls,3,TRUE
...
```

### Example 2: With Sample IDs from Betas

If your betas.RData has meaningful row names:

```bash
# Include sample IDs from betas.RData
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv \
    betas.RData
```

**Output shows actual sample names:**
```csv
sample_id,original_label,merged_label,class_id,pb_merged
GSM1234567,AML_t_PML-RARA,AML_t_PML-RARA,0,FALSE
GSM1234568,AML_t_RUNX1-RUNX1T1,AML_t_RUNX1-RUNX1T1,1,FALSE
...
```

### Example 3: Full Workflow

```bash
# 1. Export labels for inspection
Rscript convert_rdata_to_csv.R y_to_csv \
    y.RData \
    labels.csv \
    betas.RData

# 2. Inspect labels
head labels.csv

# 3. Analyze in Python
python -c "
import pandas as pd
df = pd.read_csv('labels.csv')
print('Total samples:', len(df))
print('\nOriginal classes:', df['original_label'].nunique())
print(df['original_label'].value_counts())
print('\nAfter merging:', df['merged_label'].nunique())
print(df['merged_label'].value_counts())
print('\nPB controls merged:', df['pb_merged'].sum(), 'samples')
"

# 4. Convert full training data
Rscript convert_rdata_to_csv.R marlin_to_csv \
    betas.RData \
    y.RData \
    training_data.csv
```

## Analyzing Labels in R

```r
# Load labels CSV
labels <- read.csv("labels.csv")

# View structure
str(labels)

# Original class distribution
table(labels$original_label)

# After merging
table(labels$merged_label)

# Which samples were merged?
pb_samples <- labels[labels$pb_merged == TRUE, ]
nrow(pb_samples)  # Number of PB controls merged

# Class mapping
class_map <- unique(labels[, c("original_label", "merged_label", "class_id")])
class_map[order(class_map$class_id), ]
```

## Analyzing Labels in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load labels
df = pd.read_csv('labels.csv')

# Basic statistics
print(f"Total samples: {len(df)}")
print(f"Original classes: {df['original_label'].nunique()}")
print(f"Merged classes: {df['merged_label'].nunique()}")
print(f"PB controls merged: {df['pb_merged'].sum()}")

# Class distribution (original)
print("\nOriginal label distribution:")
print(df['original_label'].value_counts().head(10))

# Class distribution (after merging)
print("\nMerged label distribution:")
print(df['merged_label'].value_counts().head(10))

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original labels
df['original_label'].value_counts().head(20).plot(kind='barh', ax=axes[0])
axes[0].set_title('Original Labels (Top 20)')
axes[0].set_xlabel('Count')

# Merged labels
df['merged_label'].value_counts().head(20).plot(kind='barh', ax=axes[1])
axes[1].set_title('Merged Labels (Top 20)')
axes[1].set_xlabel('Count')

plt.tight_layout()
plt.savefig('label_distribution.png')
print("\nVisualization saved to label_distribution.png")

# Create class mapping reference
class_mapping = df[['original_label', 'merged_label', 'class_id']].drop_duplicates()
class_mapping = class_mapping.sort_values('class_id')
class_mapping.to_csv('class_mapping_reference.csv', index=False)
print("\nClass mapping saved to class_mapping_reference.csv")

# Find which original labels were merged
merged_labels = df[df['pb_merged'] == True]['original_label'].unique()
print(f"\nLabels that were merged to 'PB controls': {len(merged_labels)}")
print(merged_labels)
```

## Use Cases

### Use Case 1: Quality Check Before Training

```bash
# 1. Export labels
Rscript convert_rdata_to_csv.R y_to_csv y.RData labels.csv

# 2. Check for issues
python check_labels.py labels.csv
```

**check_labels.py:**
```python
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

print("Label Quality Check")
print("=" * 50)

# Check for missing values
if df.isnull().any().any():
    print("⚠️ WARNING: Missing values detected!")
    print(df.isnull().sum())
else:
    print("✓ No missing values")

# Check class balance
counts = df['merged_label'].value_counts()
min_count = counts.min()
max_count = counts.max()
imbalance_ratio = max_count / min_count

print(f"\n✓ Total samples: {len(df)}")
print(f"✓ Number of classes: {df['merged_label'].nunique()}")
print(f"✓ Class balance ratio: {imbalance_ratio:.2f}x")
print(f"  Min samples per class: {min_count}")
print(f"  Max samples per class: {max_count}")

if imbalance_ratio > 10:
    print("⚠️ WARNING: Significant class imbalance detected!")
    print("   Consider using class weights or upsampling")

# List classes with few samples
low_count_classes = counts[counts < 10]
if len(low_count_classes) > 0:
    print(f"\n⚠️ Classes with <10 samples:")
    for class_name, count in low_count_classes.items():
        print(f"  {class_name}: {count} samples")
```

### Use Case 2: Create Custom Label Mapping

```python
import pandas as pd

# Load labels
df = pd.read_csv('labels.csv')

# Create custom groupings
def custom_grouping(label):
    if label.startswith('AML'):
        return 'AML_group'
    elif label.startswith('ALL'):
        return 'ALL_group'
    elif label == 'PB controls':
        return 'controls'
    else:
        return 'other'

df['custom_group'] = df['merged_label'].apply(custom_grouping)

# Save custom mapping
df.to_csv('labels_with_custom_groups.csv', index=False)

print("Custom grouping distribution:")
print(df['custom_group'].value_counts())
```

## Output Columns Explained

| Column | Description | Example Values |
|--------|-------------|----------------|
| `sample_id` | Sample identifier | GSM1234567, sample_1 |
| `original_label` | Original label from y.RData | AML_t_PML-RARA, PB_1 |
| `merged_label` | After PB merging | AML_t_PML-RARA, PB controls |
| `class_id` | 0-based numeric ID | 0, 1, 2, ... |
| `pb_merged` | Was this sample merged? | TRUE, FALSE |

## Tips

1. **Always inspect labels first** before training to understand your data
2. **Check for class imbalance** - may need upsampling
3. **Verify PB merging** - ensure all PB* labels are combined
4. **Create visualizations** to understand class distribution
5. **Keep labels.csv as reference** for interpreting predictions

## Troubleshooting

**Q: Why do I have sample_1, sample_2 instead of real IDs?**
A: Provide betas.RData as the 4th argument to get actual sample IDs

**Q: Can I skip PB merging?**
A: Currently automatic. To disable, edit the R function and set `merge_pb_controls = FALSE`

**Q: How do I map predictions back to original labels?**
A: Use the labels.csv as a lookup table with the class_id column
