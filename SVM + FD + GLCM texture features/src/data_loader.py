
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SVM'))
from feature_extraction import read_labeled_csv, extract_dataset, extract_image_features
