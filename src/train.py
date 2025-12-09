"""
MARLIN Training Script
Train the MARLIN model on methylation data using PyTorch
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import MARLINModel
from data_utils import (
    load_csv_data,
    load_reference_features,
    upsample_classes,
    create_data_loaders,
    save_class_mapping
)
from data_utils_fast import load_training_data

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=50, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (no softmax for CrossEntropyLoss)
        outputs = model(data, apply_softmax=False)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data, apply_softmax=False)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def train_marlin(
    train_csv,
    output_dir,
    reference_features=None,
    epochs=3000,
    batch_size=32,
    learning_rate=1e-5,
    upsample=True,
    samples_per_class=50,
    test_size=0.2,
    val_size=0.1,
    early_stopping_patience=100,
    save_every=100,
    device='cuda',
    random_seed=42
):
    """
    Train MARLIN model.

    Args:
        train_csv (str): Path to training data CSV file
        output_dir (str): Directory to save model and logs
        reference_features (str, optional): Path to reference features file
        epochs (int): Number of training epochs. Default: 3000
        batch_size (int): Batch size. Default: 32
        learning_rate (float): Learning rate. Default: 1e-5
        upsample (bool): Whether to upsample classes. Default: True
        samples_per_class (int): Samples per class after upsampling. Default: 50
        test_size (float): Proportion for test set. Default: 0.2
        val_size (float): Proportion for validation set. Default: 0.1
        early_stopping_patience (int): Patience for early stopping. Default: 100
        save_every (int): Save checkpoint every N epochs. Default: 100
        device (str): Device to train on ('cuda' or 'cpu'). Default: 'cuda'
        random_seed (int): Random seed for reproducibility. Default: 42
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load reference features if provided
    feature_columns = None
    if reference_features:
        feature_columns = load_reference_features(reference_features)

    # Load data (auto-detect format: CSV, HDF5, or NPZ)
    file_ext = os.path.splitext(train_csv)[1].lower()
    if file_ext in ['.h5', '.hdf5', '.npz']:
        # Fast loading for HDF5/NPZ
        print(f"Detected fast format: {file_ext}")
        data, labels, _feature_names = load_training_data(
            train_csv,
            format='auto',
            binarize=True
        )
    else:
        # CSV loading (slower but compatible)
        print("Loading from CSV format (slow)...")
        data, labels, _feature_names = load_csv_data(
            train_csv,
            feature_columns=feature_columns,
            binarize=True
        )

    # Save class mapping
    save_class_mapping(labels, os.path.join(output_dir, 'class_mapping.csv'))

    # Upsample classes if needed
    if upsample:
        data, labels = upsample_classes(
            data, labels,
            samples_per_class=samples_per_class,
            random_state=random_seed
        )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data, labels,
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_seed
    )

    # Initialize model
    input_size = data.shape[1]
    output_size = int(len(np.unique(labels)))

    model = MARLINModel(
        input_size=input_size,
        output_size=output_size
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model(os.path.join(output_dir, 'best_model.pt'))

        # Save checkpoint periodically
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            model.save_model(checkpoint_path)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch}')
            # Restore best model
            model.load_state_dict(early_stopping.best_model_state)
            break

    # Training complete
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)')

    # Final evaluation on test set
    print('\nEvaluating on test set...')
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%')

    # Save final model
    model.save_model(os.path.join(output_dir, 'final_model.pt'))

    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save training configuration
    config = {
        'train_csv': train_csv,
        'epochs': epoch,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'input_size': input_size,
        'output_size': output_size,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': total_time,
        'random_seed': random_seed
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    writer.close()

    print(f'\nModel and logs saved to {output_dir}')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

    return model, training_history


def main():
    parser = argparse.ArgumentParser(
        description='Train MARLIN model for leukemia subtype classification'
    )

    parser.add_argument(
        '--train_csv',
        type=str,
        required=True,
        help='Path to training data CSV file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save model and logs (default: ./output)'
    )

    parser.add_argument(
        '--reference_features',
        type=str,
        default=None,
        help='Path to reference features file (optional)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3000,
        help='Number of training epochs (default: 3000)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )

    parser.add_argument(
        '--no_upsample',
        action='store_true',
        help='Disable class upsampling'
    )

    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=50,
        help='Samples per class after upsampling (default: 50)'
    )

    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=100,
        help='Patience for early stopping (default: 100)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: cuda)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Train model
    train_marlin(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        reference_features=args.reference_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        upsample=not args.no_upsample,
        samples_per_class=args.samples_per_class,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
