
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXAMPLE_DIR=ROOT/'data/example_data'
DATA_DIR=Path(os.environ.get('AGS_DATA_DIR',str(EXAMPLE_DIR))).expanduser().resolve()
IMAGE_DIR=DATA_DIR
CSV_PATH=DATA_DIR/'data_paper_method.csv'
SOURCE_CSV=DATA_DIR/'data.csv'

def require_experiment_data(df):
    if len(df)!=231:
        raise ValueError(f'The current dataset contains {len(df)} records; the public examples only demonstrate the input format and preprocessing. The complete paper experiment requires 231 records. Set AGS_DATA_DIR to the local complete dataset directory.')
