

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (CLASS_NAMES, CONFUSION_MATRIX_PATH, CSV_PATH,
                    FEW_SHOT_PATH, IMAGE_DIR, MAX_NEW_TOKENS, MAX_PIXELS,
                    METRICS_PATH, MIN_PIXELS, MODEL_DIR, OFFLOAD_DIR,
                    PREDICTIONS_PATH, RANDOM_SEED, RESULTS_DIR,
                    STAGE_DESCRIPTIONS)
from data_loader import (choose_few_shot_examples, load_rgb_image,
                         read_dataset)
from model import QwenFewShotClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metrics(results, output_dir):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = results.loc[results['true_stage'].isin(CLASS_NAMES)].copy()
    if evaluated.empty:
        raise ValueError('No labeled predictions are available for metric calculation.')

    y_true = evaluated['true_stage'].tolist()
    y_pred = evaluated['predicted_stage'].fillna('invalid').tolist()
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=list(CLASS_NAMES),
        average='macro', zero_division=0,
    )
    valid_count = int(evaluated['predicted_stage'].isin(CLASS_NAMES).sum())
    report = classification_report(
        y_true, y_pred, labels=list(CLASS_NAMES),
        target_names=list(CLASS_NAMES), zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    pd.DataFrame(
        matrix, index=CLASS_NAMES, columns=CLASS_NAMES,
    ).to_csv(output_dir / CONFUSION_MATRIX_PATH.name, encoding='utf-8-sig')

    summary = (
        'Model: Qwen/Qwen3.5-4B\n'
        'Training: none\n'
        'Prompting: 9-shot (three labeled images per class; morphology guidance)\n'
        f'Evaluated images: {len(evaluated)}\n'
        f'Valid parsed predictions: {valid_count}/{len(evaluated)}\n'
        f'Accuracy: {accuracy:.6f}\n'
        f'Macro-F1: {macro_f1:.6f}\n\n'
        f'{report}'
    )
    if 'is_primary_few_shot_example' in evaluated.columns:
        strict = evaluated.loc[~evaluated['is_primary_few_shot_example'].astype(bool)]
        strict_accuracy = accuracy_score(
            strict['true_stage'], strict['predicted_stage']
        ) if not strict.empty else float('nan')
        strict_macro_f1 = f1_score(
            strict['true_stage'], strict['predicted_stage'],
            labels=list(CLASS_NAMES), average='macro', zero_division=0,
        ) if not strict.empty else float('nan')
        summary += (
            '\nStrict evaluation excluding the 9 primary few-shot images:\n'
            f'Strict evaluated images: {len(strict)}\n'
            f'Strict Accuracy: {strict_accuracy:.6f}\n'
            f'Strict Macro-F1: {strict_macro_f1:.6f}\n'
        )
    (output_dir / METRICS_PATH.name).write_text(summary, encoding='utf-8')
    print('\n' + summary)
    return accuracy, macro_f1


def upsert_prediction(results, row):
    
    if results.empty:
        return pd.DataFrame([row])
    results = results.loc[results['image'].ne(row['image'])].copy()
    return pd.concat([results, pd.DataFrame([row])], ignore_index=True)


def run(args):
    set_seed(args.seed)
    all_data = read_dataset(args.csv_path, args.image_dir)
    data = all_data
    if args.max_images is not None:
        data = data.iloc[:args.max_images].copy()
    print(f'The CSV contains {len(data)} images for inference; the model is not trained.')
    print('Class distribution:')
    print(data['stage'].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_PATH.name

    primary_examples = choose_few_shot_examples(all_data, example_config=args.example_config)
    primary_example_names = {example['image'] for example in primary_examples}
    pd.DataFrame([{
        'image': example['image'],
        'stage': example['stage'],
        'reactor_id': example['reactor_id'],
        'morphology_reference': STAGE_DESCRIPTIONS[example['stage']],
    } for example in primary_examples]).to_csv(
        output_dir / FEW_SHOT_PATH.name, index=False, encoding='utf-8-sig'
    )
    print('The default 9 few-shot examples (3 per class):')
    for example in primary_examples:
        print(f"  {example['stage']} / {example['reactor_id']}: {example['image']}")

    if predictions_path.is_file() and not args.restart:
        results = pd.read_csv(predictions_path, encoding='utf-8-sig')
        results = results.loc[results['image'].isin(set(data['image']))].copy()
        expected_examples = {
            name: '|'.join(example['image'] for example in
                           choose_few_shot_examples(all_data, query_image=name, example_config=args.example_config))
            for name in data['image']
        }
        if 'few_shot_images' not in results.columns:
            raise ValueError('The checkpoint file lacks example information; rerun with --restart.')
        if not results['few_shot_images'].fillna('').eq(
                results['image'].map(expected_examples)).where(
                    results['status'].eq('ok'), True).all():
            raise ValueError('The examples in the checkpoint file do not match the current 9-shot configuration; use --restart.')
        completed = set(results.loc[results['status'].eq('ok'), 'image'])
        print(f'A checkpoint file was detected; skipping {len(completed)} successfully completed images.')
    else:
        results = pd.DataFrame()
        completed = set()

    classifier = QwenFewShotClassifier(
        model_path=args.model_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        offload_dir=args.offload_dir,
    )

    for row in tqdm(data.itertuples(index=False), total=len(data), desc='Per-image inference'):
        if row.image in completed:
            continue
        started = time.perf_counter()
        try:
            example_rows = choose_few_shot_examples(
                all_data, query_image=row.image, example_config=args.example_config
            )
            examples = [
                (load_rgb_image(Path(args.image_dir) / example['image']),
                 example['stage'])
                for example in example_rows
            ]
            query_image = load_rgb_image(Path(args.image_dir) / row.image)
            prediction, raw_output = classifier.classify(
                query_image=query_image,
                examples=examples,
                max_new_tokens=args.max_new_tokens,
            )
            status = 'ok'
            error = ''
            example_names = '|'.join(example['image'] for example in example_rows)
        except Exception as exc:  # Save progress even when a single image fails to support troubleshooting and resuming.
            prediction = 'invalid'
            raw_output = ''
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
            example_names = ''
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {
            'image': row.image,
            'reactor_id': row.reactor_id,
            'true_stage': row.stage,
            'predicted_stage': prediction,
            'correct': prediction == row.stage,
            'status': status,
            'raw_output': raw_output,
            'error': error,
            'few_shot_images': example_names,
            'is_primary_few_shot_example': row.image in primary_example_names,
            'elapsed_seconds': time.perf_counter() - started,
        }
        results = upsert_prediction(results, result)
        order = {name: index for index, name in enumerate(data['image'])}
        results['_order'] = results['image'].map(order)
        results = results.sort_values('_order').drop(columns='_order')
        results.to_csv(predictions_path, index=False, encoding='utf-8-sig')

    if 'is_primary_few_shot_example' not in results.columns:
        results['is_primary_few_shot_example'] = results['image'].isin(
            primary_example_names
        )
        results.to_csv(predictions_path, index=False, encoding='utf-8-sig')
    save_metrics(results, output_dir)
    print(f'Per-image results: {predictions_path}')
    print(f'Metrics report: {output_dir / METRICS_PATH.name}')
    print(f'Confusion matrix: {output_dir / CONFUSION_MATRIX_PATH.name}')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Qwen3.5-4B 9-shot AGS microscopic image classification (no training)'
    )
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--csv-path', type=Path, default=CSV_PATH)
    parser.add_argument('--image-dir', type=Path, default=IMAGE_DIR)
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--offload-dir', type=Path, default=OFFLOAD_DIR)
    parser.add_argument('--min-pixels', type=int, default=MIN_PIXELS)
    parser.add_argument('--max-pixels', type=int, default=MAX_PIXELS)
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--max-images', type=int, default=None,
                        help='Debug only the first N images; do not set this option for formal evaluation.')
    parser.add_argument('--restart', action='store_true',
                        help='Ignore an existing predictions.csv and restart from the first image.')
    parser.add_argument('--example-config', type=Path, default=None)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
