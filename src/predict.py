"""
MARLIN Prediction Script
Make predictions on new methylation data using trained MARLIN model
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import MARLINModel
from data_utils import load_csv_data, load_reference_features, MARLINDataset


def predict_batch(
    model_path,
    input_csv,
    output_csv,
    reference_features=None,
    class_mapping=None,
    batch_size=32,
    device='cuda',
    return_probabilities=True
):
    """
    Make predictions on a batch of samples.

    Args:
        model_path (str): Path to trained model file (.pt)
        input_csv (str): Path to input data CSV file
        output_csv (str): Path to save predictions CSV
        reference_features (str, optional): Path to reference features file
        class_mapping (str, optional): Path to class mapping CSV file
        batch_size (int): Batch size for prediction. Default: 32
        device (str): Device to use ('cuda' or 'cpu'). Default: 'cuda'
        return_probabilities (bool): Return full probability distribution. Default: True

    Returns:
        pd.DataFrame: Predictions dataframe
    """
    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = MARLINModel.load_model(model_path, device=device)
    model.eval()

    # Load reference features if provided
    feature_columns = None
    if reference_features:
        feature_columns = load_reference_features(reference_features)

    # Load input data (no labels required for prediction)
    print(f"Loading input data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Check if labels are present
    has_labels = 'label' in df.columns
    if has_labels:
        sample_ids = df.index.tolist()
        labels_df = df[['label']]
        feature_df = df.drop(columns=['label'])
    else:
        sample_ids = df.index.tolist()
        feature_df = df

    # Check for sample ID column
    if 'sample_id' in feature_df.columns:
        sample_ids = feature_df['sample_id'].tolist()
        feature_df = feature_df.drop(columns=['sample_id'])

    # Select features if reference provided
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(feature_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} reference features not found in input")
        available_cols = [col for col in feature_columns if col in feature_df.columns]
        feature_df = feature_df[available_cols]

    # Convert to numpy and handle missing values
    data = feature_df.values
    data = np.nan_to_num(data, nan=0.0)

    # Binarize
    print("Binarizing methylation values...")
    data = np.where(data >= 0.5, 1, -1).astype(np.float32)

    print(f"Input data shape: {data.shape}")

    # Create dataset and loader
    # Use dummy labels for DataLoader (not used in prediction)
    dummy_labels = np.zeros(len(data), dtype=np.int64)
    dataset = MARLINDataset(data, dummy_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make predictions
    print("Making predictions...")
    all_predictions = []
    all_probabilities = []

    start_time = time.time()

    with torch.no_grad():
        for batch_data, _ in loader:
            batch_data = batch_data.to(device)

            # Get predictions
            outputs = model.predict_proba(batch_data)

            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.append(outputs.cpu().numpy())

    prediction_time = time.time() - start_time
    print(f"Predictions completed in {prediction_time:.2f}s "
          f"({prediction_time/len(data)*1000:.1f}ms per sample)")

    # Combine probabilities
    all_probabilities = np.vstack(all_probabilities)

    # Load class mapping if provided
    class_names = None
    if class_mapping and os.path.exists(class_mapping):
        class_map_df = pd.read_csv(class_mapping)
        class_names = class_map_df['class_name'].tolist()
        print(f"Loaded {len(class_names)} class names from mapping")

    # Create results dataframe
    results = pd.DataFrame()
    results['sample_id'] = sample_ids
    results['predicted_class'] = all_predictions

    if class_names:
        results['predicted_class_name'] = [
            class_names[idx] if idx < len(class_names) else f'Class_{idx}'
            for idx in all_predictions
        ]

    # Add true labels if available
    if has_labels:
        results['true_label'] = labels_df['label'].values

    # Add probabilities if requested
    if return_probabilities:
        if class_names:
            prob_columns = [f'prob_{name}' for name in class_names]
        else:
            prob_columns = [f'prob_class_{i}' for i in range(all_probabilities.shape[1])]

        prob_df = pd.DataFrame(all_probabilities, columns=prob_columns)
        results = pd.concat([results, prob_df], axis=1)

    # Add prediction confidence (max probability)
    results['confidence'] = all_probabilities.max(axis=1)

    # Save results
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Calculate accuracy if labels available
    if has_labels:
        # Try to match labels to predictions
        if class_names:
            # Convert true labels to indices
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}
            true_indices = [
                label_to_idx.get(label, -1)
                for label in labels_df['label'].values
            ]
            valid_mask = np.array(true_indices) >= 0
            if valid_mask.sum() > 0:
                accuracy = np.mean(
                    np.array(all_predictions)[valid_mask] == np.array(true_indices)[valid_mask]
                )
                print(f"Accuracy: {accuracy*100:.2f}% ({valid_mask.sum()}/{len(valid_mask)} samples)")

    # Print prediction summary
    print("\nPrediction Summary:")
    if class_names:
        pred_counts = pd.Series([class_names[p] for p in all_predictions]).value_counts()
    else:
        pred_counts = pd.Series(all_predictions).value_counts()
    print(pred_counts.head(10))
    print(f"\nMean confidence: {results['confidence'].mean():.4f}")
    print(f"Median confidence: {results['confidence'].median():.4f}")

    return results


def predict_single(
    model_path,
    input_csv,
    reference_features=None,
    class_mapping=None,
    device='cuda'
):
    """
    Make prediction on a single sample and return detailed results.

    Args:
        model_path (str): Path to trained model file (.pt)
        input_csv (str): Path to input data CSV file (single sample)
        reference_features (str, optional): Path to reference features file
        class_mapping (str, optional): Path to class mapping CSV file
        device (str): Device to use ('cuda' or 'cpu'). Default: 'cuda'

    Returns:
        dict: Prediction results with top classes and probabilities
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MARLINModel.load_model(model_path, device=device)
    model.eval()

    # Load reference features if provided
    feature_columns = None
    if reference_features:
        feature_columns = load_reference_features(reference_features)

    # Load data
    data, _, _ = load_csv_data(
        input_csv,
        feature_columns=feature_columns,
        label_column=None if 'label' not in pd.read_csv(input_csv).columns else 'label',
        binarize=True
    )

    if len(data) != 1:
        raise ValueError(f"Expected single sample, got {len(data)} samples")

    # Convert to tensor
    data_tensor = torch.FloatTensor(data).to(device)

    # Predict
    with torch.no_grad():
        probabilities = model.predict_proba(data_tensor)[0].cpu().numpy()

    # Get top predictions
    top_k = 5
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_probs = probabilities[top_indices]

    # Load class names if available
    class_names = None
    if class_mapping and os.path.exists(class_mapping):
        class_map_df = pd.read_csv(class_mapping)
        class_names = class_map_df['class_name'].tolist()

    # Prepare results
    results = {
        'predicted_class': int(top_indices[0]),
        'confidence': float(top_probs[0]),
        'top_predictions': []
    }

    if class_names:
        results['predicted_class_name'] = class_names[top_indices[0]]

    for idx, prob in zip(top_indices, top_probs):
        pred = {
            'class_idx': int(idx),
            'probability': float(prob)
        }
        if class_names and idx < len(class_names):
            pred['class_name'] = class_names[idx]
        results['top_predictions'].append(pred)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained MARLIN model'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model file (.pt)'
    )

    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to input data CSV file'
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        default='predictions.csv',
        help='Path to save predictions CSV (default: predictions.csv)'
    )

    parser.add_argument(
        '--reference_features',
        type=str,
        default=None,
        help='Path to reference features file (optional)'
    )

    parser.add_argument(
        '--class_mapping',
        type=str,
        default=None,
        help='Path to class mapping CSV file (optional)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for prediction (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    parser.add_argument(
        '--no_probabilities',
        action='store_true',
        help='Do not include full probability distribution in output'
    )

    args = parser.parse_args()

    # Make predictions
    predict_batch(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        reference_features=args.reference_features,
        class_mapping=args.class_mapping,
        batch_size=args.batch_size,
        device=args.device,
        return_probabilities=not args.no_probabilities
    )


if __name__ == '__main__':
    main()
