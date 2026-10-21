

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
from pathlib import Path
import sys
REPOSITORY_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'dataset_paths.py').is_file())
sys.path.insert(0, str(REPOSITORY_ROOT))
from dataset_paths import DATA_DIR, IMAGE_DIR, CSV_PATH

IMAGE_DIR = DATA_DIR

MODEL_ID = 'Qwen/Qwen3.5-4B'
MODEL_DIR = PROJECT_DIR / 'models' / 'Qwen3.5-4B'
OFFLOAD_DIR = PROJECT_DIR / 'models' / 'offload'

RESULTS_DIR = PROJECT_DIR / 'results' / 'few_shot_9'
PREDICTIONS_PATH = RESULTS_DIR / 'predictions.csv'
METRICS_PATH = RESULTS_DIR / 'metrics.txt'
CONFUSION_MATRIX_PATH = RESULTS_DIR / 'confusion_matrix.csv'
FEW_SHOT_PATH = RESULTS_DIR / 'few_shot_examples.csv'

CLASS_NAMES = ('growth', 'stabilization', 'disintegration')
EXAMPLES_PER_CLASS = 3
REACTOR_IDS = ('Z1', 'Z2', 'Z3')
# Manually selected from morphological previews of labeled images; the order corresponds to Z1, Z2, and Z3.
PRIMARY_EXAMPLE_IMAGES = {
    'growth': ('image24.png', 'image180.jpg', 'image65.png'),
    'stabilization': ('image130.png', 'image35.png', 'image102.png'),
    'disintegration': ('image87.png', 'image125.png', 'image30.png'),
}
STAGE_DESCRIPTIONS = {
    'growth': 'Granule outlines gradually form, the structure becomes more compact, and boundaries are relatively intact, showing aggregation and granule formation.',
    'stabilization': 'Granule sizes and outlines are relatively stable, with compact structures, clear and intact boundaries, and few broken or free fragments.',
    'disintegration': 'Granule edges are loose, filamentous structures extend outward, surrounding fragments increase, boundaries are damaged, and granules disintegrate.',
}
RANDOM_SEED = 42

# 10 images (9 examples + 1 query) enter the context together. Limit visual resolution to control GPU memory use.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MAX_NEW_TOKENS = 24

