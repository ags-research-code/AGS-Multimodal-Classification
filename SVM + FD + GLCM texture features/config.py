
from pathlib import Path
from pathlib import Path
import sys
REPOSITORY_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'dataset_paths.py').is_file())
sys.path.insert(0, str(REPOSITORY_ROOT))
from dataset_paths import DATA_DIR, IMAGE_DIR, CSV_PATH

IMAGE_DIR=DATA_DIR
RESULTS_DIR=Path(__file__).resolve().parent/'SVM/results'
BASE_SEED=42
NUM_REPEATS=5
VALIDATION_FRACTION=0.2
C=10
GAMMA='scale'
KERNEL='rbf'
GLCM_LEVELS=32
INPUT_FEATURES=('FD','mean_intensity','std_intensity','contrast','correlation','energy','homogeneity')
