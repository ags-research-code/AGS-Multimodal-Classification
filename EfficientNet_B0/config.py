import os
import torch

# Data paths: all data are stored in the data directory one level above the project folder
from pathlib import Path
import sys
REPOSITORY_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'dataset_paths.py').is_file())
sys.path.insert(0, str(REPOSITORY_ROOT))
from dataset_paths import DATA_DIR, IMAGE_DIR, CSV_PATH


# Model parameters - retain the original ResNet34 training budget for a fair comparison
BATCH_SIZE = 12  # Table S3
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100  # Table S3: fixed at 100 epochs
NUM_REPEATS = 5  # Number of repetitions of the complete three-fold experiment
BASE_SEED = 42  # seed = 42 + 100*(repeat-1) + (fold-1)
VALIDATION_FRACTION = 0.2  # Split the other two reactors 80:20 into training/validation sets
WEIGHT_DECAY = 1e-4  # Weight decay: the smaller value (1e-4) balances regularization and feature learning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select the compute device automatically
NUM_CLASSES = 3  # Number of classes for classification: adjust to the actual dataset

# Model save path
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'saved')

# Results save path
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

