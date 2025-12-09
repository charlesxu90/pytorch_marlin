"""
MARLIN PyTorch Model Architecture
Methylation- and AI-guided Rapid Leukemia Subtype Inference

Neural Network Architecture:
- Input: 357,340 CpG sites (binarized to +1/-1)
- Hidden Layer 1: 256 nodes (sigmoid activation) with 99% dropout
- Hidden Layer 2: 128 nodes (sigmoid activation)
- Output: 42 leukemia subtypes (softmax activation)
"""

import torch
import torch.nn as nn


class MARLINModel(nn.Module):
    """
    MARLIN deep learning model for acute leukemia classification.

    Args:
        input_size (int): Number of input features (CpG sites). Default: 357340
        hidden1_size (int): Size of first hidden layer. Default: 256
        hidden2_size (int): Size of second hidden layer. Default: 128
        output_size (int): Number of output classes. Default: 42
        dropout_rate (float): Dropout rate for regularization. Default: 0.99
    """

    def __init__(
        self,
        input_size=357340,
        hidden1_size=256,
        hidden2_size=128,
        output_size=42,
        dropout_rate=0.99
    ):
        super(MARLINModel, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Dropout on INPUT (matching Keras: layer_dropout comes first!)
        self.dropout_input = nn.Dropout(p=dropout_rate)

        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.sigmoid1 = nn.Sigmoid()

        # Second hidden layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.sigmoid2 = nn.Sigmoid()

        # Output layer
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, apply_softmax=True):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            apply_softmax (bool): Whether to apply softmax to output.
                                 Set to False during training with CrossEntropyLoss

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
        """
        # Dropout on INPUT features (matching R: dropout BEFORE first dense layer)
        x = self.dropout_input(x)

        # First hidden layer
        x = self.fc1(x)
        x = self.sigmoid1(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.sigmoid2(x)

        # Output layer
        x = self.fc3(x)

        # Apply softmax if needed (for inference)
        if apply_softmax:
            x = self.softmax(x)

        return x

    def predict_proba(self, x):
        """
        Predict class probabilities (inference mode).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Probability distribution of shape (batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, apply_softmax=True)

    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, filepath):
        """
        Save model weights and architecture configuration.

        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate
        }, filepath)

    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """
        Load model from saved checkpoint.

        Args:
            filepath (str): Path to the saved model
            device (str): Device to load the model on ('cpu' or 'cuda')

        Returns:
            MARLINModel: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            input_size=checkpoint['input_size'],
            hidden1_size=checkpoint['hidden1_size'],
            hidden2_size=checkpoint['hidden2_size'],
            output_size=checkpoint['output_size'],
            dropout_rate=checkpoint['dropout_rate']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model


if __name__ == '__main__':
    # Test model instantiation
    model = MARLINModel()
    print(f"MARLIN Model Architecture:")
    print(f"  Input size: {model.input_size:,}")
    print(f"  Hidden layer 1: {model.hidden1_size}")
    print(f"  Hidden layer 2: {model.hidden2_size}")
    print(f"  Output size: {model.output_size}")
    print(f"  Dropout rate: {model.dropout_rate}")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, model.input_size)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output probabilities sum: {output.sum(dim=1)[0].item():.4f}")
