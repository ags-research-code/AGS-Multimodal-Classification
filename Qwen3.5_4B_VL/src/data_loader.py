

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (CLASS_NAMES, CSV_PATH, EXAMPLES_PER_CLASS, IMAGE_DIR,
                    PRIMARY_EXAMPLE_IMAGES, REACTOR_IDS)  # noqa: E402


COLUMN_NAMES = ['image', 'fd', 'd50', 'feature3', 'stage', 'reactor_id']


def read_dataset(csv_path=CSV_PATH, image_dir=IMAGE_DIR):
    
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    data = None
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            data = pd.read_csv(
                csv_path, header=None, encoding=encoding,
                dtype=str, keep_default_na=False,
            )
            break
        except UnicodeDecodeError:
            continue
    if data is None or data.empty:
        raise ValueError('The CSV could not be read or is empty.')
    if data.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f'The CSV must have 6 columns; found {data.shape[1]} columns: {csv_path}'
        )
    if str(data.iloc[0, 0]).strip().lower() in {
            'image', 'image_name', 'filename', 'file_name'}:
        data = data.iloc[1:].copy()

    data.columns = COLUMN_NAMES
    data = data.reset_index(drop=True)
    for column in ('image', 'stage', 'reactor_id'):
        data[column] = data[column].astype(str).str.strip()
    data['stage'] = data['stage'].str.lower()
    data['reactor_id'] = data['reactor_id'].str.upper()

    unknown = sorted(set(data['stage']) - set(CLASS_NAMES))
    missing_classes = sorted(set(CLASS_NAMES) - set(data['stage']))
    if unknown or missing_classes:
        raise ValueError(
            f'Classes do not match expectations; unknown classes={unknown}, missing classes={missing_classes}'
        )
    if data['image'].eq('').any() or data['image'].duplicated().any():
        raise ValueError('The CSV contains empty or duplicate image filenames.')
    missing_images = [
        name for name in data['image'] if not (image_dir / name).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f'Missing {len(missing_images)} images, for example: {missing_images[:5]}'
        )
    return data


def choose_few_shot_examples(data, query_image=None, example_config=None):
    
    selected_config = PRIMARY_EXAMPLE_IMAGES
    if example_config is not None:
        import json
        selected_config = json.loads(Path(example_config).read_text(encoding="utf-8"))
    examples = []
    for label in CLASS_NAMES:
        candidates = data.loc[data['stage'].eq(label)]
        candidates = candidates.loc[candidates['image'].ne(query_image)]
        if len(candidates) < EXAMPLES_PER_CLASS:
            raise ValueError(f'Class {label} has fewer than {EXAMPLES_PER_CLASS} examples after excluding the query image; the 10 public images only demonstrate the format. Nine-shot evaluation requires at least 4 images per class. Specify the complete local dataset.')
        selected = []
        for reactor, preferred in zip(REACTOR_IDS, selected_config[label]):
            pool = candidates.loc[candidates['reactor_id'].eq(reactor)]
            if pool.empty:
                continue
            primary = pool.loc[pool['image'].eq(preferred)]
            selected.append((primary if not primary.empty else pool).iloc[0])
        selected_names = {row['image'] for row in selected}
        remaining = candidates.loc[~candidates['image'].isin(selected_names)]
        for _, row in remaining.iterrows():
            if len(selected) >= EXAMPLES_PER_CLASS:
                break
            selected.append(row)
        examples.extend(selected)
    return examples


def load_rgb_image(path):
    
    with Image.open(path) as image:
        return image.convert('RGB')
