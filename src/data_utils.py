"""
MARLIN Data Preprocessing Utilities
Handles data loading, binarization, and dataset preparation
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List


class MARLINDataset(Dataset):
    """
    PyTorch Dataset for MARLIN methylation data.

    Args:
        data (np.ndarray): Methylation data of shape (n_samples, n_features)
        labels (np.ndarray): Class labels of shape (n_samples,)
        transform (callable, optional): Optional transform to be applied on a sample
    """

    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def binarize_methylation(beta_values, threshold=0.5):
    """
    Binarize methylation beta values to +1/-1.

    Args:
        beta_values (np.ndarray): Beta values in range [0, 1]
        threshold (float): Threshold for binarization. Default: 0.5

    Returns:
        np.ndarray: Binarized values (+1 for methylated, -1 for unmethylated)
    """
    binarized = np.where(beta_values >= threshold, 1, -1)
    return binarized.astype(np.float32)


def load_csv_data(
    csv_path,
    feature_columns=None,
    label_column='label',
    binarize=True,
    threshold=0.5
):
    """
    Load methylation data from CSV file.

    CSV Format:
        - First column (or specified label_column): class labels
        - Remaining columns: methylation beta values for CpG sites
        - Column names should match feature reference if provided

    Args:
        csv_path (str): Path to CSV file
        feature_columns (list, optional): List of feature column names to use
        label_column (str): Name of the label column. Default: 'label'
        binarize (bool): Whether to binarize the data. Default: True
        threshold (float): Binarization threshold. Default: 0.5

    Returns:
        tuple: (data, labels, feature_names)
            - data (np.ndarray): Methylation data of shape (n_samples, n_features)
            - labels (np.ndarray): Class labels of shape (n_samples,)
            - feature_names (list): List of feature column names
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract labels
    if label_column in df.columns:
        labels = df[label_column].values
        feature_df = df.drop(columns=[label_column])
    else:
        raise ValueError(f"Label column '{label_column}' not found in CSV")

    # Select features if specified
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(feature_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features not found in CSV")
        available_cols = [col for col in feature_columns if col in feature_df.columns]
        feature_df = feature_df[available_cols]

    feature_names = feature_df.columns.tolist()
    data = feature_df.values

    # Handle missing values (replace with 0)
    data = np.nan_to_num(data, nan=0.0)

    # Binarize if requested
    if binarize:
        print("Binarizing methylation values...")
        data = binarize_methylation(data, threshold=threshold)

    # Convert string labels to integers if needed
    if labels.dtype == object or labels.dtype.kind == 'U':
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_idx[label] for label in labels])
        print(f"Converted {len(unique_labels)} unique class labels to indices")

    print(f"Loaded data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(np.unique(labels))}")

    return data, labels, feature_names


def load_reference_features(reference_path):
    """
    Load reference CpG feature names.

    Args:
        reference_path (str): Path to CSV/TXT file with feature names
                             (one feature per line or single column CSV)

    Returns:
        list: List of feature names
    """
    if reference_path.endswith('.csv'):
        df = pd.read_csv(reference_path)
        features = df.iloc[:, 0].tolist()
    else:
        with open(reference_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(features)} reference features from {reference_path}")
    return features


def upsample_classes(data, labels, samples_per_class=50, random_state=42):
    """
    Upsample minority classes to ensure equal representation.

    Args:
        data (np.ndarray): Methylation data of shape (n_samples, n_features)
        labels (np.ndarray): Class labels of shape (n_samples,)
        samples_per_class (int): Target number of samples per class. Default: 50
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (upsampled_data, upsampled_labels)
    """
    np.random.seed(random_state)

    unique_classes = np.unique(labels)
    upsampled_data = []
    upsampled_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        cls_data = data[cls_indices]

        # If we have fewer samples than target, upsample with replacement
        if len(cls_indices) < samples_per_class:
            indices = np.random.choice(
                len(cls_indices),
                size=samples_per_class,
                replace=True
            )
            cls_data = cls_data[indices]
        # If we have more samples, randomly sample without replacement
        elif len(cls_indices) > samples_per_class:
            indices = np.random.choice(
                len(cls_indices),
                size=samples_per_class,
                replace=False
            )
            cls_data = cls_data[indices]

        upsampled_data.append(cls_data)
        upsampled_labels.extend([cls] * samples_per_class)

    upsampled_data = np.vstack(upsampled_data)
    upsampled_labels = np.array(upsampled_labels)

    # Shuffle the data
    shuffle_indices = np.random.permutation(len(upsampled_labels))
    upsampled_data = upsampled_data[shuffle_indices]
    upsampled_labels = upsampled_labels[shuffle_indices]

    print(f"Upsampled data: {len(unique_classes)} classes, "
          f"{samples_per_class} samples each, total: {len(upsampled_labels)}")

    return upsampled_data, upsampled_labels


def create_data_loaders(
    data,
    labels,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=0
):
    """
    Create train, validation, and test data loaders.

    Args:
        data (np.ndarray): Methylation data
        labels (np.ndarray): Class labels
        batch_size (int): Batch size for training. Default: 32
        test_size (float): Proportion of data for testing. Default: 0.2
        val_size (float): Proportion of training data for validation. Default: 0.1
        random_state (int): Random seed for reproducibility
        num_workers (int): Number of workers for data loading. Default: 0

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train_val
    )

    # Create datasets
    train_dataset = MARLINDataset(X_train, y_train)
    val_dataset = MARLINDataset(X_val, y_val)
    test_dataset = MARLINDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Data split: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


def save_class_mapping(labels, output_path):
    """
    Save mapping between class indices and names.

    Args:
        labels (np.ndarray or list): Original class labels (strings or ints)
        output_path (str): Path to save the mapping CSV
    """
    unique_labels = sorted(set(labels))
    mapping_df = pd.DataFrame({
        'class_idx': range(len(unique_labels)),
        'class_name': unique_labels
    })
    mapping_df.to_csv(output_path, index=False)
    print(f"Saved class mapping to {output_path}")


if __name__ == '__main__':
    # Example usage
    print("MARLIN Data Utilities - Example Usage\n")

    # Example: Create dummy data
    print("Creating dummy data...")
    n_samples = 100
    n_features = 357340
    n_classes = 42

    dummy_data = np.random.rand(n_samples, n_features)
    dummy_labels = np.random.randint(0, n_classes, n_samples)

    print(f"Dummy data shape: {dummy_data.shape}")
    print(f"Dummy labels shape: {dummy_labels.shape}")

    # Test binarization
    print("\nTesting binarization...")
    binarized = binarize_methylation(dummy_data)
    print(f"Unique values after binarization: {np.unique(binarized)}")

    # Test upsampling
    print("\nTesting upsampling...")
    upsampled_data, upsampled_labels = upsample_classes(
        dummy_data, dummy_labels, samples_per_class=10
    )

    # Test data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        upsampled_data, upsampled_labels, batch_size=32
    )

    print(f"\nData loader batch sizes:")
    for batch_data, batch_labels in train_loader:
        print(f"  Train batch: data={batch_data.shape}, labels={batch_labels.shape}")
        break
