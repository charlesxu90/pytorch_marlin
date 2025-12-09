# MARLIN PyTorch model

**Key Features:**
- PyTorch-based neural network (357,340 → 256 → 128 → 42)
- ⚡ **Fast HDF5 format (10-100x faster than CSV!)**
- CSV data format support
- Batch training and prediction
- Early stopping and model checkpointing
- TensorBoard logging
- GPU acceleration support

## Architecture

```
Input Layer:     357,340 CpG sites (binarized to +1/-1)
                    ↓
Dropout Layer:   99% dropout (extreme regularization)
                    ↓
Hidden Layer 1:  256 nodes (sigmoid activation)
                    ↓
Hidden Layer 2:  128 nodes (sigmoid activation)
                    ↓
Output Layer:    42 leukemia subtypes (softmax)
```

**Total Parameters:** ~91.5 million


## Directory Structure

```
pytorch_marlin/
├── model.py              # MARLIN model architecture
├── data_utils.py         # Data loading and preprocessing utilities
├── train.py              # Training script
├── predict.py            # Prediction script
├── convert_data.py       # Data conversion utilities
└── README.md            # This file
```



### 2. Train the Model

```bash
python src/train_exact.py --train_file data/training_data.h5
```

**Training Options:**
- `--train_csv`: Path to training data CSV
- `--output_dir`: Directory to save model and logs
- `--reference_features`: Path to reference features file (optional)
- `--epochs`: Number of training epochs (default: 3000)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--samples_per_class`: Samples per class after upsampling (default: 50)
- `--no_upsample`: Disable class upsampling
- `--early_stopping_patience`: Patience for early stopping (default: 100)
- `--device`: Device to train on (cuda/cpu)
- `--seed`: Random seed for reproducibility (default: 42)

### 3. Make Predictions

```bash
python predict.py \
    --model_path ./output/best_model.pt \
    --input_csv test_data.csv \
    --output_csv predictions.csv \
    --reference_features reference_features.txt \
    --class_mapping ./output/class_mapping.csv \
    --batch_size 32 \
    --device cuda
```

**Prediction Options:**
- `--model_path`: Path to trained model (.pt file)
- `--input_csv`: Path to input data CSV
- `--output_csv`: Path to save predictions
- `--reference_features`: Path to reference features file (optional)
- `--class_mapping`: Path to class mapping CSV (optional)
- `--batch_size`: Batch size for prediction (default: 32)
- `--no_probabilities`: Don't include full probability distribution in output
- `--device`: Device to use (cuda/cpu)