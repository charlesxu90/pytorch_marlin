"""
MARLIN Training Script - Exact R Implementation
Matches the original MARLIN_training.R exactly
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from model import MARLINModel
from data_utils_fast import load_training_data


class MARLINDataset(Dataset):
    """PyTorch Dataset for MARLIN."""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def flip_x_percent(x, x_percent=0.1, random_state=None):
    """
    Flip x% of elements in a vector (data augmentation).
    Matches R function: flip_x_percent from MARLIN_training.R line 61-74

    Args:
        x (np.ndarray): Input vector
        x_percent (float): Percentage to flip (0-1)
        random_state: Random state for reproducibility

    Returns:
        np.ndarray: Vector with flipped elements
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_flip = int(x_percent * len(x))
    sample_indices = np.random.choice(len(x), size=n_flip, replace=False)

    x_copy = x.copy()
    x_copy[sample_indices] = -x_copy[sample_indices]

    return x_copy


def upsample_and_augment(data, labels, samples_per_class=50, flip_percent=0.1, random_seed=100):
    """
    Upsample classes and apply data augmentation.
    Matches R code lines 79-88.

    Args:
        data (np.ndarray): Methylation data
        labels (np.ndarray): Class labels
        samples_per_class (int): Samples per class (default: 50)
        flip_percent (float): Percentage of CpGs to flip (default: 0.1)
        random_seed (int): Random seed (default: 100 to match R)

    Returns:
        tuple: (upsampled_data, upsampled_labels)
    """
    np.random.seed(random_seed)

    unique_classes = np.unique(labels)
    upsampled_data = []
    upsampled_labels = []

    print(f"Upsampling to {samples_per_class} samples per class...")

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]

        # Sample with replacement (matching R behavior)
        sampled_indices = np.random.choice(
            cls_indices,
            size=samples_per_class,
            replace=True
        )

        cls_data = data[sampled_indices]
        upsampled_data.append(cls_data)
        upsampled_labels.extend([cls] * samples_per_class)

    upsampled_data = np.vstack(upsampled_data)
    upsampled_labels = np.array(upsampled_labels)

    print(f"Upsampled: {len(unique_classes)} classes × {samples_per_class} = {len(upsampled_labels)} samples")

    # Apply data augmentation: flip 10% of CpGs
    if flip_percent > 0:
        print(f"Applying data augmentation: flipping {flip_percent*100}% of CpGs...")
        np.random.seed(random_seed)

        for i in range(len(upsampled_data)):
            upsampled_data[i] = flip_x_percent(upsampled_data[i], flip_percent)

        print("Data augmentation complete")

    return upsampled_data, upsampled_labels


def train_marlin_exact(
    train_file,
    output_dir,
    epochs=3000,
    batch_size=32,
    learning_rate=1e-5,
    samples_per_class=50,
    flip_percent=0.1,
    device='cuda',
    random_seed=100
):
    """
    Train MARLIN model matching the exact R implementation.

    Args:
        train_file (str): Path to training data (.h5 or .csv)
        output_dir (str): Output directory
        epochs (int): Number of epochs (default: 3000)
        batch_size (int): Batch size (default: 32)
        learning_rate (float): Learning rate (default: 1e-5)
        samples_per_class (int): Samples per class for upsampling (default: 50)
        flip_percent (float): Percentage of CpGs to flip (default: 0.1)
        device (str): Device ('cuda' or 'cpu')
        random_seed (int): Random seed (default: 100 to match R)
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {train_file}...")
    data, labels, feature_names = load_training_data(
        train_file,
        format='auto',
        binarize=True
    )

    print(f"Loaded: {data.shape[0]} samples × {data.shape[1]} features")
    print(f"Classes: {len(np.unique(labels))}")

    # Upsample and augment (matching R code exactly)
    data, labels = upsample_and_augment(
        data, labels,
        samples_per_class=samples_per_class,
        flip_percent=flip_percent,
        random_seed=random_seed
    )

    # Create dataset and loader (NO train/val/test split - train on everything like R!)
    dataset = MARLINDataset(data, labels)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Matching R: shuffle = TRUE
        num_workers=0
    )

    print(f"\nTraining on ALL {len(dataset)} samples (no validation split)")
    print(f"Batches per epoch: {len(train_loader)}")

    # Initialize model
    input_size = data.shape[1]
    output_size = len(np.unique(labels))

    model = MARLINModel(
        input_size=input_size,
        output_size=output_size,
        dropout_rate=0.99  # Matching R
    ).to(device)

    print(f"\nModel: {input_size} → 256 → 128 → {output_size}")
    print(f"Dropout: 0.99")
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Loss and optimizer (matching R exactly)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Collect predictions and labels for precision/recall
        all_predictions = []
        all_labels = []

        for data_batch, labels_batch in train_loader:
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data_batch, apply_softmax=False)
            loss = criterion(outputs, labels_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * data_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            # Collect for precision/recall
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        # Calculate precision, recall, and F1 score (weighted average for multi-class)
        epoch_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Print progress (matching R style output with precision/recall/f1)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}/{epochs} - '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Accuracy: {epoch_acc:.2f}%, '
                  f'Precision: {epoch_precision:.4f}, '
                  f'Recall: {epoch_recall:.4f}, '
                  f'F1: {epoch_f1:.4f} '
                  f'({elapsed:.1f}s)')

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            model.save_model(checkpoint_path)

    # Save final model
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)')

    final_model_path = os.path.join(output_dir, 'marlin_model.pt')
    model.save_model(final_model_path)
    print(f'Model saved to {final_model_path}')

    # Final evaluation on training data
    print('\nEvaluating final model...')
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data_batch, labels_batch in train_loader:
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
            outputs = model(data_batch, apply_softmax=False)
            loss = criterion(outputs, labels_batch)

            running_loss += loss.item() * data_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    final_loss = running_loss / total
    final_accuracy = round(100.0 * correct / total, 4)
    final_precision = round(precision_score(all_labels, all_predictions, average='weighted', zero_division=0), 4)
    final_recall = round(recall_score(all_labels, all_predictions, average='weighted', zero_division=0), 4)
    final_f1 = round(f1_score(all_labels, all_predictions, average='weighted', zero_division=0), 4)

    # Save training configuration
    config = {
        'train_file': train_file,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'samples_per_class': samples_per_class,
        'flip_percent': flip_percent,
        'input_size': input_size,
        'output_size': output_size,
        'final_loss': round(final_loss, 4),
        'final_accuracy': final_accuracy,
        'final_precision': final_precision,
        'final_recall': final_recall,
        'final_f1_score': final_f1,
        'training_time': round(total_time, 4),
        'random_seed': random_seed
    }

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'Config saved to {config_path}')
    print(f'\nFinal Metrics:')
    print(f'  Accuracy:  {final_accuracy:.4f}%')
    print(f'  Precision: {final_precision:.4f}')
    print(f'  Recall:    {final_recall:.4f}')
    print(f'  F1 Score:  {final_f1:.4f}')

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train MARLIN model (exact R implementation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script exactly matches the original MARLIN_training.R implementation:
  - Trains on ALL upsampled data (no validation split)
  - Applies 10% CpG flipping for data augmentation
  - Uses random seed 100 (same as R)
  - Uses same hyperparameters (batch=32, lr=1e-5, dropout=0.99)

Example:
  python train_exact.py --train_file training_data.h5 --output_dir ./output
        '''
    )

    parser.add_argument('--train_file', required=True,
                        help='Path to training data (.h5 or .csv)')
    parser.add_argument('--output_dir', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs (default: 3000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='Samples per class (default: 50)')
    parser.add_argument('--flip_percent', type=float, default=0.1,
                        help='Percentage of CpGs to flip (default: 0.1)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=100,
                        help='Random seed (default: 100 to match R)')

    args = parser.parse_args()

    train_marlin_exact(
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        samples_per_class=args.samples_per_class,
        flip_percent=args.flip_percent,
        device=args.device,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
