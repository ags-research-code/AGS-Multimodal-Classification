
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from feature_extraction import CLASSES


def macro_f1_scorer(estimator, x, y):
    
    return f1_score(y, estimator.predict(x), labels=CLASSES,
                    average='macro', zero_division=0)


def analyze(results_dir, n_repeats=30):
    if n_repeats < 2:
        raise ValueError('At least 2 permutation repetitions are required.')
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob('repeat_*/fold_*/svm_model.joblib'))
    if not paths:
        paths = sorted(results_dir.glob('repeat_*/svm_model.joblib'))
    if len(paths)!=15:
        raise FileNotFoundError('A complete set of 5 repetitions × 3 folds, totaling 15 models, is required. Run train_svm.py first.')
    summaries, raw_scores = [], []
    for path in paths:
        saved = joblib.load(path)
        if saved.get('protocol')!='loro_independent_80_20_v1':raise ValueError('Only independent leave-one-reactor-out models are supported.')
        repeat_id=saved.get('repeat_id',int((path.parent.parent if path.parent.name.startswith('fold_') else path.parent).name.split('_')[-1]))
        fold_id=saved.get('fold_id',int(path.parent.name.split('_')[1]))
        if set(saved.get('train_images',[])) & set(saved.get('test_images',[])):raise ValueError('Training and test samples overlap.')
        importance = permutation_importance(saved['model'], saved['x_test'], saved['y_test'],
                                            scoring=macro_f1_scorer, n_repeats=n_repeats,
                                            random_state=saved['seed'], n_jobs=1)
        for idx, name in enumerate(saved['feature_names']):
            summaries.append({'repeat_id': repeat_id, 'seed': saved['seed'], 'fold_id': fold_id, 'feature': name,
                              'importance_mean': importance.importances_mean[idx],
                              'permutation_std': importance.importances_std[idx]})
            for permutation_id, value in enumerate(importance.importances[idx], 1):
                raw_scores.append({'repeat_id': repeat_id, 'seed': saved['seed'], 'fold_id': fold_id, 'feature': name,
                                   'permutation_id': permutation_id, 'macro_f1_drop': value})
        print(f'Completed test-set feature importance analysis for seed={saved["seed"]}, fold={saved.get("fold_id", 0)}')
    output_dir = results_dir/'feature_importance'
    output_dir.mkdir(exist_ok=True)
    table = pd.DataFrame(summaries)
    counts=table.groupby(['repeat_id','feature']).fold_id.nunique()
    if set(table.repeat_id)!={1,2,3,4,5} or not counts.eq(3).all():raise ValueError('Repetition identifiers or three-fold records are incomplete.')
    table.to_csv(output_dir/'importance_by_fold.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(raw_scores).to_csv(output_dir/'permutation_scores.csv', index=False, encoding='utf-8-sig')
    # Average the importance equally across the three folds within each repetition first, then calculate the sample standard deviation across the five repetitions.
    by_seed = table.groupby(['repeat_id', 'feature'], as_index=False).importance_mean.mean()
    by_seed.to_csv(output_dir/'importance_by_repeat.csv', index=False, encoding='utf-8-sig')
    summary = by_seed.groupby('feature').importance_mean.agg(['mean', 'std']).sort_values('mean', ascending=False)
    summary.to_csv(output_dir/'importance_summary.csv', encoding='utf-8-sig')
    plot = summary.sort_values('mean')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot.index, plot['mean']*100, xerr=plot['std']*100, capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Decrease in test Macro-F1 (percentage points)')
    ax.set_title('Permutation importance: mean ± SD across 5 repeats')
    fig.tight_layout()
    fig.savefig(output_dir/'feature_importance.png', dpi=200)
    plt.close(fig)
    note = ('Importance is the decrease in Macro-F1 on the original test set after permuting one feature.\n'
            'Only the independent test reactor from each fold is used; training and validation samples from that fold are excluded.\n'
            'Average the three fold-specific importance values equally within each repetition, then calculate the mean/sample standard deviation across five repetitions.\n'
            'The averaged values are decreases in fold-specific Macro-F1, rather than decreases in Macro-F1 pooled across 231 images.\n'
            'permutation_std: population standard deviation across permutations within the same model (ddof=0).\n'
            'Negative values are retained and indicate an increase in score after permutation. Correlated features may substitute for one another; importance does not represent a causal contribution.\n'
            'Importance on the original test set is used only for post hoc interpretation, not for parameter tuning or feature selection.\n')
    (output_dir/'README.txt').write_text(note, encoding='utf-8')
    print(summary.to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--n-repeats', type=int, default=30)
    args = parser.parse_args()
    analyze(args.results_dir, args.n_repeats)


if __name__ == '__main__':
    main()
