"""
MARLIN Example Workflow
Demonstrates complete training and prediction pipeline
"""

import os
import numpy as np
import pandas as pd
from train import train_marlin
from predict import predict_batch, predict_single


def create_dummy_data(n_samples=100, n_features=1000, n_classes=10, output_file='dummy_data.csv'):
    """
    Create dummy methylation data for testing.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of CpG features
        n_classes (int): Number of classes
        output_file (str): Output CSV file path

    Returns:
        str: Path to created CSV file
    """
    print(f"Creating dummy data: {n_samples} samples × {n_features} features, {n_classes} classes")

    # Generate random beta values (0-1)
    data = np.random.rand(n_samples, n_features)

    # Generate random labels
    labels = np.random.randint(0, n_classes, n_samples)
    label_names = [f'Class_{i}' for i in range(n_classes)]
    labels = [label_names[i] for i in labels]

    # Create feature names
    feature_names = [f'cg{i:08d}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df.insert(0, 'label', labels)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dummy data saved to {output_file}")

    return output_file


def example_training():
    """
    Example: Train MARLIN model on dummy data.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Training MARLIN Model")
    print("="*80 + "\n")

    # Create dummy training data
    train_csv = create_dummy_data(
        n_samples=200,
        n_features=1000,  # Reduced for demo (real: 357340)
        n_classes=10,     # Reduced for demo (real: 42)
        output_file='dummy_train.csv'
    )

    # Train model
    print("\nTraining model...")
    output_dir = './example_output'

    model, history = train_marlin(
        train_csv=train_csv,
        output_dir=output_dir,
        epochs=50,           # Reduced for demo (real: 3000)
        batch_size=16,
        learning_rate=1e-4,
        upsample=True,
        samples_per_class=20,  # Reduced for demo (real: 50)
        early_stopping_patience=10,
        save_every=25,
        device='cpu',        # Use 'cuda' if GPU available
        random_seed=42
    )

    print(f"\nTraining completed!")
    print(f"Model saved to {output_dir}/best_model.pt")
    print(f"Training history saved to {output_dir}/training_history.json")

    return output_dir


def example_prediction(model_dir):
    """
    Example: Make predictions using trained model.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Making Predictions")
    print("="*80 + "\n")

    # Create dummy test data
    test_csv = create_dummy_data(
        n_samples=20,
        n_features=1000,
        n_classes=10,
        output_file='dummy_test.csv'
    )

    # Make predictions
    print("\nMaking predictions...")
    output_csv = 'example_predictions.csv'

    results = predict_batch(
        model_path=f'{model_dir}/best_model.pt',
        input_csv=test_csv,
        output_csv=output_csv,
        class_mapping=f'{model_dir}/class_mapping.csv',
        batch_size=16,
        device='cpu',
        return_probabilities=True
    )

    print(f"\nPredictions saved to {output_csv}")
    print(f"\nPrediction Results (first 5 samples):")
    print(results.head())

    # Example single sample prediction
    print("\n" + "="*80)
    print("EXAMPLE 3: Single Sample Prediction")
    print("="*80 + "\n")

    # Create single sample
    single_sample_csv = 'dummy_single.csv'
    df = pd.read_csv(test_csv)
    df.iloc[:1].to_csv(single_sample_csv, index=False)

    result = predict_single(
        model_path=f'{model_dir}/best_model.pt',
        input_csv=single_sample_csv,
        class_mapping=f'{model_dir}/class_mapping.csv',
        device='cpu'
    )

    print(f"Predicted Class: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nTop 5 Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"  {i}. {pred['class_name']}: {pred['probability']:.4f}")

    return results


def example_model_usage():
    """
    Example: Direct model usage in Python.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Direct Model Usage")
    print("="*80 + "\n")

    import torch
    from model import MARLINModel

    # Create a small model for demonstration
    model = MARLINModel(
        input_size=1000,
        hidden1_size=128,
        hidden2_size=64,
        output_size=10,
        dropout_rate=0.5
    )

    print("Model Architecture:")
    print(f"  Input size: {model.input_size:,}")
    print(f"  Hidden layer 1: {model.hidden1_size}")
    print(f"  Hidden layer 2: {model.hidden2_size}")
    print(f"  Output size: {model.output_size}")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Create dummy input (batch of 5 samples)
    batch_size = 5
    dummy_input = torch.randn(batch_size, model.input_size)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model.predict_proba(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"\nPredicted classes: {predictions.argmax(dim=1).numpy()}")
    print(f"Prediction confidences: {predictions.max(dim=1)[0].numpy()}")

    # Save and load model
    model.save_model('example_model.pt')
    print(f"\nModel saved to example_model.pt")

    loaded_model = MARLINModel.load_model('example_model.pt', device='cpu')
    print(f"Model loaded successfully")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("MARLIN PyTorch Implementation - Example Workflow")
    print("="*80)

    # Example 1: Training
    model_dir = example_training()

    # Example 2 & 3: Prediction
    example_prediction(model_dir)

    # Example 4: Direct model usage
    example_model_usage()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")

    print("Generated files:")
    print("  - dummy_train.csv          (training data)")
    print("  - dummy_test.csv           (test data)")
    print("  - dummy_single.csv         (single sample)")
    print("  - example_output/          (trained model and logs)")
    print("  - example_predictions.csv  (prediction results)")
    print("  - example_model.pt         (demo model)")

    print("\nNext steps:")
    print("  1. Replace dummy data with your real methylation data")
    print("  2. Adjust model parameters (epochs, batch_size, etc.)")
    print("  3. Use GPU (--device cuda) for faster training")
    print("  4. Monitor training with TensorBoard:")
    print("     tensorboard --logdir example_output/logs")


if __name__ == '__main__':
    main()
