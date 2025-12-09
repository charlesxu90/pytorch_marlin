"""
MARLIN Data Conversion Script
Convert BED files and R data to CSV format for PyTorch training
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def convert_bed_to_csv(bed_file, output_csv=None):
    """
    Convert BED file to CSV format.

    BED format (tab-separated):
    chr  start  end  beta_value  probe_id

    CSV format:
    probe_id (or chr:start-end), beta_value

    Args:
        bed_file (str): Path to input BED file
        output_csv (str, optional): Path to output CSV. If None, uses bed_file with .csv extension

    Returns:
        pd.DataFrame: Converted data
    """
    print(f"Reading BED file: {bed_file}")

    # Read BED file (handle both .bed and .bed.gz)
    if bed_file.endswith('.gz'):
        df = pd.read_csv(bed_file, sep='\t', compression='gzip', header=None)
    else:
        df = pd.read_csv(bed_file, sep='\t', header=None)

    # Standard BED format: chr, start, end, name, score, strand
    # MARLIN format appears to be: chr, pos, beta_value, probe_id
    if df.shape[1] == 4:
        df.columns = ['chr', 'pos', 'beta_value', 'probe_id']
    elif df.shape[1] == 5:
        df.columns = ['chr', 'start', 'end', 'beta_value', 'probe_id']
    else:
        print(f"Warning: Unexpected number of columns ({df.shape[1]}). Using default names.")
        df.columns = [f'col_{i}' for i in range(df.shape[1])]

    # Create probe identifier if not present
    if 'probe_id' not in df.columns:
        if 'start' in df.columns and 'end' in df.columns:
            df['probe_id'] = df['chr'].astype(str) + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
        elif 'pos' in df.columns:
            df['probe_id'] = df['chr'].astype(str) + ':' + df['pos'].astype(str)

    # Extract relevant columns
    if 'beta_value' in df.columns and 'probe_id' in df.columns:
        result = df[['probe_id', 'beta_value']]
    else:
        print(f"Warning: Could not find expected columns. Available: {df.columns.tolist()}")
        result = df

    # Save to CSV
    if output_csv is None:
        output_csv = bed_file.replace('.bed.gz', '.csv').replace('.bed', '.csv')

    result.to_csv(output_csv, index=False)
    print(f"Saved to CSV: {output_csv} ({len(result)} probes)")

    return result


def convert_multiple_beds_to_matrix(
    bed_files,
    output_csv,
    sample_labels=None,
    reference_features=None
):
    """
    Convert multiple BED files to a single training matrix CSV.

    Args:
        bed_files (list): List of BED file paths
        output_csv (str): Path to output CSV file
        sample_labels (list, optional): List of class labels for each sample
        reference_features (str, optional): Path to reference features file to standardize probes

    Returns:
        pd.DataFrame: Training matrix
    """
    print(f"Converting {len(bed_files)} BED files to training matrix...")

    # Load reference features if provided
    if reference_features:
        print(f"Loading reference features from {reference_features}")
        if reference_features.endswith('.csv'):
            ref_df = pd.read_csv(reference_features)
            reference_probes = ref_df.iloc[:, 0].tolist()
        else:
            with open(reference_features, 'r') as f:
                reference_probes = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(reference_probes)} reference probes")
    else:
        reference_probes = None

    # Process each BED file
    sample_data = []
    sample_names = []

    for i, bed_file in enumerate(bed_files):
        print(f"Processing {i+1}/{len(bed_files)}: {bed_file}")

        # Read BED file
        if bed_file.endswith('.gz'):
            df = pd.read_csv(bed_file, sep='\t', compression='gzip', header=None)
        else:
            df = pd.read_csv(bed_file, sep='\t', header=None)

        # Parse BED format
        if df.shape[1] == 4:
            df.columns = ['chr', 'pos', 'beta_value', 'probe_id']
        elif df.shape[1] == 5:
            df.columns = ['chr', 'start', 'end', 'beta_value', 'probe_id']

        # Create probe ID if needed
        if 'probe_id' not in df.columns:
            if 'pos' in df.columns:
                df['probe_id'] = df['chr'].astype(str) + ':' + df['pos'].astype(str)

        # Get sample name
        sample_name = Path(bed_file).stem.replace('.bed', '')
        sample_names.append(sample_name)

        # Create series with probe_id as index
        probe_series = pd.Series(
            df['beta_value'].values,
            index=df['probe_id'].values
        )

        sample_data.append(probe_series)

    # Combine all samples
    print("Combining samples into matrix...")
    data_matrix = pd.DataFrame(sample_data, index=sample_names)

    # Fill missing values with 0
    data_matrix = data_matrix.fillna(0)

    # Align to reference features if provided
    if reference_probes:
        print("Aligning to reference features...")
        # Keep only reference probes that exist in data
        available_probes = [p for p in reference_probes if p in data_matrix.columns]
        missing_probes = [p for p in reference_probes if p not in data_matrix.columns]

        print(f"  Available: {len(available_probes)} / {len(reference_probes)}")
        print(f"  Missing: {len(missing_probes)}")

        # Reorder and add missing columns
        data_matrix = data_matrix.reindex(columns=reference_probes, fill_value=0)

    # Add labels if provided
    if sample_labels:
        if len(sample_labels) != len(sample_names):
            raise ValueError(f"Number of labels ({len(sample_labels)}) does not match "
                           f"number of samples ({len(sample_names)})")
        data_matrix.insert(0, 'label', sample_labels)

    # Save to CSV
    print(f"Saving training matrix to {output_csv}...")
    data_matrix.to_csv(output_csv, index=False)

    print(f"Training matrix saved: {data_matrix.shape[0]} samples × {data_matrix.shape[1]} features")

    return data_matrix


def create_reference_features_from_bed(bed_file, output_file):
    """
    Extract probe IDs from a reference BED file.

    Args:
        bed_file (str): Path to reference BED file
        output_file (str): Path to save feature names

    Returns:
        list: List of feature names
    """
    print(f"Extracting features from {bed_file}...")

    if bed_file.endswith('.gz'):
        df = pd.read_csv(bed_file, sep='\t', compression='gzip', header=None)
    else:
        df = pd.read_csv(bed_file, sep='\t', header=None)

    # Assume last column is probe ID
    if df.shape[1] == 4:
        features = df.iloc[:, 3].tolist()
    elif df.shape[1] == 5:
        features = df.iloc[:, 4].tolist()
    else:
        # Create probe IDs from coordinates
        features = (df.iloc[:, 0].astype(str) + ':' +
                   df.iloc[:, 1].astype(str)).tolist()

    # Save features
    with open(output_file, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")

    print(f"Saved {len(features)} features to {output_file}")

    return features


def main():
    parser = argparse.ArgumentParser(
        description='Convert MARLIN data files to CSV format'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['bed_to_csv', 'beds_to_matrix', 'extract_features'],
        help='Conversion mode'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file or directory (for beds_to_matrix, use glob pattern)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file'
    )

    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='CSV file with sample labels (sample_id,label) for beds_to_matrix mode'
    )

    parser.add_argument(
        '--reference_features',
        type=str,
        default=None,
        help='Reference features file for beds_to_matrix mode'
    )

    args = parser.parse_args()

    if args.mode == 'bed_to_csv':
        convert_bed_to_csv(args.input, args.output)

    elif args.mode == 'beds_to_matrix':
        # Find BED files
        bed_files = sorted(glob.glob(args.input))
        print(f"Found {len(bed_files)} BED files")

        if len(bed_files) == 0:
            print(f"Error: No BED files found matching pattern: {args.input}")
            return

        # Load labels if provided
        sample_labels = None
        if args.labels:
            labels_df = pd.read_csv(args.labels)
            # Assumes CSV has columns: sample_id, label
            # Match by filename
            label_dict = dict(zip(labels_df['sample_id'], labels_df['label']))
            sample_labels = []
            for bed_file in bed_files:
                sample_id = Path(bed_file).stem.replace('.bed', '')
                sample_labels.append(label_dict.get(sample_id, 'unknown'))

        convert_multiple_beds_to_matrix(
            bed_files,
            args.output,
            sample_labels=sample_labels,
            reference_features=args.reference_features
        )

    elif args.mode == 'extract_features':
        create_reference_features_from_bed(args.input, args.output)


if __name__ == '__main__':
    main()
