"""
MARLIN Fast Data Loading Utilities
Optimized for large methylation datasets using HDF5/NPZ formats
"""

import os
import numpy as np
import h5py


def load_data_hdf5(hdf5_file, return_feature_names=False):
    """
    Load data from HDF5 format (very fast).

    Args:
        hdf5_file (str): Path to HDF5 file
        return_feature_names (bool): Whether to return feature names

    Returns:
        tuple: (data, labels) or (data, labels, feature_names)
    """
    print(f"Loading from HDF5: {hdf5_file}")

    with h5py.File(hdf5_file, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]

        # Convert labels if they're bytes
        if labels.dtype.kind == 'S':
            labels = labels.astype(str)

        print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")

        if return_feature_names:
            feature_names = f['feature_names'][:].astype(str)
            return data, labels, feature_names
        else:
            return data, labels


def load_data_npz(npz_file, return_feature_names=False):
    """
    Load data from NumPy NPZ format.

    Args:
        npz_file (str): Path to NPZ file
        return_feature_names (bool): Whether to return feature names

    Returns:
        tuple: (data, labels) or (data, labels, feature_names)
    """
    print(f"Loading from NPZ: {npz_file}")

    npz_data = np.load(npz_file, allow_pickle=True)
    data = npz_data['data']
    labels = npz_data['labels']

    print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")

    if return_feature_names:
        feature_names = npz_data['feature_names']
        return data, labels, feature_names
    else:
        return data, labels


def save_data_hdf5(data, labels, feature_names, output_file, compression='gzip'):
    """
    Save data to HDF5 format (10-100x faster loading than CSV).

    Args:
        data (np.ndarray): Methylation data (n_samples, n_features)
        labels (np.ndarray): Class labels (n_samples,)
        feature_names (list): Feature names
        output_file (str): Output HDF5 file path
        compression (str): Compression algorithm ('gzip', 'lzf', or None)
    """
    print(f"Saving to HDF5: {output_file}")
    print(f"  Data shape: {data.shape}")
    print(f"  Compression: {compression}")

    with h5py.File(output_file, 'w') as f:
        # Save data with compression
        f.create_dataset('data', data=data, compression=compression, compression_opts=4 if compression == 'gzip' else None)

        # Save labels
        if labels.dtype.kind in ['U', 'S', 'O']:  # String labels
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('labels', data=labels.astype(str), dtype=dt)
        else:  # Numeric labels
            f.create_dataset('labels', data=labels)

        # Save feature names
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('feature_names', data=[str(name) for name in feature_names], dtype=dt)

        # Save metadata
        metadata = f.create_group('metadata')
        metadata.attrs['n_samples'] = data.shape[0]
        metadata.attrs['n_features'] = data.shape[1]
        metadata.attrs['n_classes'] = len(np.unique(labels))
        metadata.attrs['data_dtype'] = str(data.dtype)

    file_size = os.path.getsize(output_file) / (1024**3)  # GB
    print(f"  File size: {file_size:.2f} GB")
    print(f"Saved successfully!")


def save_data_npz(data, labels, feature_names, output_file, compressed=True):
    """
    Save data to NumPy NPZ format (fast, no external dependencies).

    Args:
        data (np.ndarray): Methylation data
        labels (np.ndarray): Class labels
        feature_names (list): Feature names
        output_file (str): Output NPZ file path
        compressed (bool): Use compression (savez_compressed)
    """
    print(f"Saving to NPZ: {output_file}")
    print(f"  Data shape: {data.shape}")
    print(f"  Compressed: {compressed}")

    save_func = np.savez_compressed if compressed else np.savez

    save_func(
        output_file,
        data=data,
        labels=labels,
        feature_names=np.array(feature_names, dtype=object)
    )

    file_size = os.path.getsize(output_file) / (1024**3)  # GB
    print(f"  File size: {file_size:.2f} GB")
    print(f"Saved successfully!")


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


def load_training_data(
    data_path,
    format='auto',
    binarize=True,
    threshold=0.5
):
    """
    Load training data from various formats (auto-detect or specify).

    Args:
        data_path (str): Path to data file (.csv, .h5, .hdf5, .npz)
        format (str): Format ('auto', 'csv', 'hdf5', 'npz')
        binarize (bool): Whether to binarize data
        threshold (float): Binarization threshold

    Returns:
        tuple: (data, labels, feature_names)
    """
    # Auto-detect format
    if format == 'auto':
        ext = os.path.splitext(data_path)[1].lower()
        if ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        elif ext == '.npz':
            format = 'npz'
        elif ext == '.csv':
            format = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    print(f"Loading data (format: {format})...")

    # Load based on format
    if format == 'hdf5':
        data, labels, feature_names = load_data_hdf5(data_path, return_feature_names=True)
    elif format == 'npz':
        data, labels, feature_names = load_data_npz(data_path, return_feature_names=True)
    elif format == 'csv':
        from data_utils import load_csv_data
        data, labels, feature_names = load_csv_data(data_path, binarize=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Convert string labels to integers if needed
    if labels.dtype == object or labels.dtype.kind in ['U', 'S']:
        print("Converting string labels to integer indices...")
        # Convert bytes to strings if needed
        if labels.dtype.kind == 'S':
            labels = labels.astype(str)

        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
        print(f"  Converted {len(unique_labels)} unique class labels to indices (0-{len(unique_labels)-1})")

    # Binarize if needed
    if binarize and format in ['hdf5', 'npz']:
        print("Binarizing methylation values...")
        data = binarize_methylation(data, threshold=threshold)

    return data, labels, feature_names
