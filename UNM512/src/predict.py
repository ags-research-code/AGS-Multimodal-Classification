
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CSV_PATH, IMAGE_DIR, MODEL_SAVE_DIR, RESULTS_DIR, DEVICE
from model import MultiModalModel


def load_inference_csv(csv_path):
    df = None
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(csv_path, header=None, encoding=encoding,
                             dtype=str, keep_default_na=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None or df.empty:
        raise ValueError('The CSV could not be read or is empty.')
    col_num = df.shape[1]
    if col_num not in (4, 5, 6):
        raise ValueError('The prediction CSV requires 4 columns (image filename + 3 features), 5 columns (plus stage), or 6 columns (plus reactor).')
    if str(df.iloc[0, 0]).strip().lower() in {'image', 'image_name', 'filename', 'file_name'}:
        df = df.iloc[1:].copy()
    df = df.reset_index(drop=True)
    cols = ['image', 'fd', 'd50', 'feature3', 'stage', 'reactor_id'][:col_num]
    df.columns = cols
    for col in ('fd', 'd50', 'feature3'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    df['image'] = df['image'].astype(str).str.strip()
    if 'reactor_id' in df:
        df['reactor_id'] = df['reactor_id'].str.upper().str.strip()
    return df


class Predictor:
    def __init__(self, model_path=None, test_reactor='Z3'):
        self.test_reactor = test_reactor
        self.results_dir = Path(RESULTS_DIR) / f'test_{test_reactor}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if model_path is None:
            fold_id = ('Z1', 'Z2', 'Z3').index(test_reactor) + 1
            model_path = (Path(MODEL_SAVE_DIR) / 'repeat_01'
                          / f'fold_{fold_id:02d}_test_{test_reactor}' / 'final_model.pth')
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Model not found: {model_path}; run train.py first.')
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        classes = checkpoint.get('classes')
        if classes is None:
            raise ValueError('The model file lacks class information in classes; retrain using the updated train.py.')
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(classes)
        self.model = MultiModalModel(num_classes=len(classes), pretrained=False).to(DEVICE)
        state = checkpoint['model_state_dict']
        if any(name.startswith('image_model.') for name in state):
            raise ValueError('These weights contain an older image branch. Retrain with the modified train.py before loading new weights.')
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, csv_path=CSV_PATH, image_dir=IMAGE_DIR, only_test_reactor=True):
        df = load_inference_csv(csv_path)
        if only_test_reactor:
            if 'reactor_id' not in df.columns:
                raise ValueError('The CSV must contain reactor_id to predict only Z3.')
            df = df[df['reactor_id'] == self.test_reactor].copy()
        if df.empty:
            raise ValueError('No images remain after CSV filtering; check the reactor identifier.')
        results = []
        with torch.inference_mode():
            for row in df.itertuples(index=False):
                data = torch.tensor([[row.fd, row.d50, row.feature3]],
                                    dtype=torch.float32, device=DEVICE)
                probs = torch.softmax(self.model(data), dim=1)[0]
                pred = int(probs.argmax().item())
                item = {'image': row.image,
                        'predicted_stage': self.label_encoder.inverse_transform([pred])[0],
                        'confidence': float(probs[pred].item())}
                if 'stage' in df.columns:
                    item['true_stage'] = row.stage
                if 'reactor_id' in df.columns:
                    item['reactor_id'] = row.reactor_id
                results.append(item)
        result_df = pd.DataFrame(results)
        out_path = self.results_dir / f'predictions_{self.test_reactor}.csv'
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'Predicted {len(result_df)} images; results saved to: {out_path}')
        return result_df

    def visualize_predictions(self, results):
        fig, ax = plt.subplots(figsize=(7, 5))
        results['predicted_stage'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel('Predicted stage')
        ax.set_ylabel('Count')
        fig.tight_layout()
        fig.savefig(self.results_dir / 'prediction_distribution.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using only FD, d50, and feature3 from the CSV')
    parser.add_argument('--csv_path', default=CSV_PATH)
    parser.add_argument('--image_dir', default=IMAGE_DIR)
    parser.add_argument('--data_dir', default=None, help='Legacy usage: the CSV and images are in the same directory')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--test-reactor', choices=['Z1', 'Z2', 'Z3'], default='Z3')
    parser.add_argument('--all', action='store_true', help='Predict all images in the CSV (default: only the held-out reactor)')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    csv_path, image_dir = args.csv_path, args.image_dir
    if args.data_dir is not None:
        image_dir = args.data_dir
        candidates = [p for p in Path(args.data_dir).glob('*.csv')
                      if not p.name.lower().startswith('predictions')]
        if len(candidates) != 1:
            raise ValueError(f'The --data_dir directory must contain exactly one input CSV; found {len(candidates)}.'
                             'Specify --csv_path and --image_dir instead.')
        csv_path = candidates[0]
    predictor = Predictor(model_path=args.model_path, test_reactor=args.test_reactor)
    results = predictor.predict(csv_path, image_dir, only_test_reactor=not args.all)
    if args.visualize:
        predictor.visualize_predictions(results)
    print(results.head())
