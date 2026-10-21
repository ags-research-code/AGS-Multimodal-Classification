
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (BATCH_SIZE, CSV_PATH, VALIDATION_FRACTION,
                    IMAGE_DIR, RESULTS_DIR)  # noqa: E402

REPOSITORY_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'protocol.py').is_file())
sys.path.insert(0, str(REPOSITORY_ROOT))
from protocol import split_reactors, save_manifest

COLUMN_NAMES = ['image', 'fd', 'd50', 'feature3', 'stage', 'reactor_id']
REACTORS = {'Z1', 'Z2', 'Z3'}


# The current default preprocessing uses only fixed Resize/ToTensor/Normalize, without random augmentation.
# For the small dataset (approximately 231 images), an in-process LRU cache avoids repeated disk reads, decoding, and transformations in each epoch.
# Note: a custom transform, especially random augmentation, bypasses this cache.
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=512)
def _cached_preprocessed_image(full_path):
    
    with Image.open(full_path) as img:
        return DEFAULT_TRANSFORM(img.convert('RGB'))


def read_labeled_csv(path):
    
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CSV not found: {path}')
    df = None
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(path, header=None, encoding=encoding, dtype=str,
                             keep_default_na=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError('Unrecognized CSV encoding; save as UTF-8 or GBK.')
    if df.shape[1] != 6:
        raise ValueError(f'The CSV must have 6 columns: image filename, 3 numerical features (FD, D50, Δt), stage label, and reactor_id; found {df.shape[1]} columns.')

    # Explicitly skip the header in CSV files exported from Excel annotation tables.
    first = str(df.iloc[0, 0]).strip().lower()
    if first in {'image', 'image_name', 'filename', 'file_name'}:
        df = df.iloc[1:].copy()
    df.columns = COLUMN_NAMES
    df = df.reset_index(drop=True)
    for col in ('image', 'stage', 'reactor_id'):
        df[col] = df[col].astype(str).str.strip()
    df['reactor_id'] = df['reactor_id'].str.upper()
    if df.empty:
        raise ValueError('The CSV is empty.')
    if df['image'].eq('').any() or df['image'].duplicated().any():
        raise ValueError('The CSV contains blank or duplicate image filenames; check the data.')
    invalid = df.loc[~df['reactor_id'].isin(REACTORS), ['image', 'reactor_id']]
    if not invalid.empty:
        raise ValueError('Some images have missing or invalid reactor identifiers (only Z1/Z2/Z3 are allowed):\n'
                         + invalid.head(8).to_string(index=False))
    if df['stage'].eq('').any():
        raise ValueError('The stage label column contains missing values.')
    return df


class ImageClassificationDataset(Dataset):
    

    def __init__(self, data_df, image_dir=IMAGE_DIR, label_encoder=None,
                 transform=None, fit_label_encoder=False):
        self.data_df = data_df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        if self.data_df.empty:
            raise ValueError('No images remain after filtering; check the reactor identifier.')
        self.use_default_cache = transform is None
        self.transform = DEFAULT_TRANSFORM if transform is None else transform
        if fit_label_encoder:
            self.label_encoder = LabelEncoder().fit(self.data_df['stage'])
        elif label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            raise ValueError('Test data must use the training label_encoder; refitting is not allowed.')
        unknown = set(self.data_df['stage']) - set(self.label_encoder.classes_)
        if unknown:
            raise ValueError(f'The test set contains classes absent from the training set: {sorted(unknown)}.')
        self.encoded_labels = self.label_encoder.transform(self.data_df['stage'])
        self.num_classes = len(self.label_encoder.classes_)
        self.image_names = self.data_df['image'].tolist()
        self.reactor_ids = self.data_df['reactor_id'].tolist()
        self.training_flags = self.data_df.get(
            'was_used_for_training', pd.Series(False, index=self.data_df.index)
        ).astype(bool).tolist()
        self.target_labels = torch.as_tensor(self.encoded_labels, dtype=torch.long)
        missing = [n for n in self.image_names
                   if not (self.image_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(
                f'Image directory {self.image_dir} is missing {len(missing)} images, for example: {missing[:5]}')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name
        if self.use_default_cache:
            image = _cached_preprocessed_image(str(image_path.resolve()))
        else:
            # Custom image augmentation is applied on each access to preserve its randomness.
            with Image.open(image_path) as img:
                image = self.transform(img.convert('RGB'))
        return {
            'images': image,
            'target': self.target_labels[idx],
            'num_classes': self.num_classes,
            'image_name': image_name,
            'reactor_id': self.reactor_ids[idx],
            'was_used_for_training': self.training_flags[idx],
        }


def get_data_loaders(test_reactor='Z3', random_seed=42,
                     results_dir=RESULTS_DIR):
    
    test_reactor = str(test_reactor).upper()
    df = read_labeled_csv(CSV_PATH)
    train_df, val_df, test_df = split_reactors(
        df, test_reactor, random_seed, VALIDATION_FRACTION)
    save_manifest(train_df, val_df, test_df, Path(results_dir) / 'split_manifest.csv')

    train_dataset = ImageClassificationDataset(train_df, fit_label_encoder=True)
    val_dataset = ImageClassificationDataset(
        val_df, label_encoder=train_dataset.label_encoder
    )
    test_dataset = ImageClassificationDataset(
        test_df, label_encoder=train_dataset.label_encoder
    )
    # Training must include all three stages; the test set may contain fewer than three classes, which is displayed explicitly during evaluation.
    if train_dataset.num_classes != 3:
        raise ValueError(f'The training set contains only {train_dataset.num_classes} classes ({list(train_dataset.label_encoder.classes_)}), '
                         'The three-class model in the original paper cannot be trained.')
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(train_dataset.label_encoder, Path(results_dir) / 'label_encoder.pkl')

    # For models with BatchNorm, avoid a final training batch containing only one image when possible.
    if len(train_dataset) < 2:
        raise ValueError('Fewer than 2 training samples are available; BatchNorm cannot train.')
    if int(BATCH_SIZE) < 2:
        raise ValueError('BATCH_SIZE must be at least 2 because the model contains BatchNorm.')
    drop_last = len(train_dataset) > BATCH_SIZE and len(train_dataset) % BATCH_SIZE == 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    print(f'Independent leave-one-reactor-out: test {test_reactor}; train {len(train_df)} images, '
          f'validation {len(val_df)} images, test {len(test_df)} images; the three sets do not overlap.')
    if drop_last:
        print('Reminder: BatchNorm and batch-size constraints cause the final random sample to be dropped in each training epoch; '
              'adjust BATCH_SIZE to avoid this.')
    return train_loader, val_loader, test_loader, train_dataset.num_classes
