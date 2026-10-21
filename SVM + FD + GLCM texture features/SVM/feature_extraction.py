
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FEATURE_NAMES = ['FD', 'mean_intensity', 'std_intensity', 'contrast',
                 'correlation', 'energy', 'homogeneity']
CLASSES = ['disintegration', 'growth', 'stabilization']


def read_labeled_csv(path):
    
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            raw = pd.read_csv(path, header=None, dtype=str,
                              keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('Unrecognized CSV encoding; use UTF-8 or GBK.')
    header = raw.iloc[0].str.strip().str.lower().tolist()
    if header[0] in {'image', 'image_name', 'filename', 'file_name'}:
        raw = raw.iloc[1:].copy()
        raw.columns = header
        raw = raw.rename(columns={'image': 'image_name', 'filename': 'image_name',
                                  'file_name': 'image_name', 'fd': 'FD', 'stage': 'label'})
    elif raw.shape[1] == 6:
        raw.columns = ['image_name', 'FD', 'd50', 'feature3', 'label', 'reactor_id']
    elif raw.shape[1] == 3:
        raw.columns = ['image_name', 'FD', 'label']
    else:
        raise ValueError('The CSV must contain image_name, FD, and label, or the six columns used in this project.')
    if not {'image_name', 'FD', 'label'}.issubset(raw.columns):
        raise ValueError('The CSV lacks image_name, FD, or label.')
    df = raw.reset_index(drop=True)
    for col in ('image_name', 'label'):
        df[col] = df[col].str.strip()
    if df.empty or df.image_name.eq('').any() or df.image_name.duplicated().any():
        raise ValueError('The CSV is empty, or image filenames are blank or duplicated.')
    df['FD'] = pd.to_numeric(df['FD'], errors='raise')
    if not np.isfinite(df.FD).all():
        raise ValueError('FD contains NaN or infinity.')
    if set(df.label) != set(CLASSES):
        raise ValueError(f'Labels must contain exactly the three classes {CLASSES}; found {sorted(set(df.label))}.')
    return df


def glcm_features(gray, levels=32):
    
    if not 2 <= levels <= 256:
        raise ValueError('The number of GLCM gray levels must be between 2 and 256.')
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 2:
        raise ValueError('GLCM input must be a uint8 grayscale image of at least 2×2 pixels.')
    quantized = gray.astype(np.uint16) * levels // 256
    i, j = np.indices((levels, levels), dtype=float)
    delta2 = (i - j) ** 2
    values = []
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
        h, w = gray.shape
        a = quantized[max(0, -dy):h-max(0, dy), max(0, -dx):w-max(0, dx)]
        b = quantized[max(0, dy):h-max(0, -dy), max(0, dx):w-max(0, -dx)]
        # Convert to int64 first to avoid pixel-pair encoding overflow when levels=256.
        codes = a.astype(np.int64).ravel() * levels + b.ravel()
        counts = np.bincount(codes, minlength=levels*levels).reshape(levels, levels)
        symmetric = counts.astype(float) + counts.T
        p = symmetric / symmetric.sum()
        mi, mj = (p*i).sum(), (p*j).sum()
        si = np.sqrt((p*(i-mi)**2).sum())
        sj = np.sqrt((p*(j-mj)**2).sum())
        correlation = 1.0 if min(si, sj) < 1e-15 else (p*(i-mi)*(j-mj)).sum()/(si*sj)
        values.append([(p*delta2).sum(), correlation, np.sqrt((p*p).sum()),
                       (p/(1+delta2)).sum()])
    return np.mean(values, axis=0)


def extract_image_features(image_path, fd, levels=32):
    
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f'Image does not exist: {image_path}')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV cannot decode the image: {image_path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.array([float(fd), float(gray.mean()), float(gray.std(ddof=0)),
                     *glcm_features(gray, levels)], dtype=float)


def extract_dataset(csv_path, image_dir, levels=32):
    df = read_labeled_csv(csv_path)
    rows = []
    for index, row in df.iterrows():
        rows.append(extract_image_features(Path(image_dir)/row.image_name, row.FD, levels))
        if (index+1) % 25 == 0 or index+1 == len(df):
            print(f'Handcrafted feature extraction: {index+1}/{len(df)}')
    x = pd.DataFrame(rows, columns=FEATURE_NAMES)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('The extracted features contain non-finite values.')
    return df, x
