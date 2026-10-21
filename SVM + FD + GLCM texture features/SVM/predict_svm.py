
import argparse
from pathlib import Path
import joblib

from feature_extraction import extract_dataset, FEATURE_NAMES
from train_svm import DATA_DIR, HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-path', '--csv_path', type=Path, default=DATA_DIR/'data_paper_method.csv')
    parser.add_argument('--image-dir', '--image_dir', type=Path, default=DATA_DIR)
    parser.add_argument('--model-path', '--model_path', type=Path,
                        default=HERE/'results/repeat_01/fold_01_test_Z1/svm_model.joblib')
    parser.add_argument('--output-path', type=Path, default=HERE/'results/inference_predictions.csv')
    args = parser.parse_args()
    saved = joblib.load(args.model_path)
    if saved['feature_names'] != FEATURE_NAMES:
        raise ValueError('The feature order in the model does not match the current extractor.')
    metadata, x = extract_dataset(args.csv_path, args.image_dir, saved['config']['glcm_levels'])
    model = saved['model']
    metadata['predicted_stage'] = model.predict(x)
    probabilities = model.predict_proba(x)
    for index, label in enumerate(model.classes_):
        metadata[f'prob_{label}'] = probabilities[:, index]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f'Predictions saved to: {args.output_path}')


if __name__ == '__main__':
    main()
