
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

from feature_extraction import CLASSES


def evaluate_model(model, x_test, y_test, metadata, output_dir, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    metrics = dict(model='SVM_FD_GLCM', seed=seed,
                   accuracy=accuracy_score(y_test, predicted),
                   precision=precision_score(y_test, predicted, labels=CLASSES, average='macro', zero_division=0),
                   recall=recall_score(y_test, predicted, labels=CLASSES, average='macro', zero_division=0),
                   macro_f1=f1_score(y_test, predicted, labels=CLASSES, average='macro', zero_division=0))
    predictions = metadata.reset_index(drop=True).copy()
    predictions['true_stage'] = list(y_test)
    predictions['predicted_stage'] = predicted
    if 'was_used_for_training' not in predictions:
        predictions['was_used_for_training'] = False
    predictions['is_correct'] = predictions.true_stage == predictions.predicted_stage
    for index, label in enumerate(model.classes_):
        predictions[f'prob_{label}'] = probabilities[:, index]
    predictions.to_csv(output_dir/'final_test_predictions.csv', index=False, encoding='utf-8-sig')
    report = classification_report(y_test, predicted, labels=CLASSES, target_names=CLASSES,
                                   digits=6, zero_division=0)
    seen = int(predictions.was_used_for_training.sum())
    text = f'Seed: {seed}\nTest samples: {len(y_test)}\nUsed for training: {seen}\n'
    if seen:
        text += 'Evaluation: existing ResNet split, including training/test overlap\n'
    for name, title in [('accuracy', 'Accuracy'), ('precision', 'Precision (macro)'),
                        ('recall', 'Recall (macro)'), ('macro_f1', 'Macro-F1')]:
        text += f'{title}: {metrics[name]:.6f}\n'
    text += '\n' + report
    (output_dir/'final_test_report.txt').write_text(text, encoding='utf-8')
    pd.DataFrame(classification_report(y_test, predicted, labels=CLASSES,
                 output_dict=True, zero_division=0)).T.to_csv(output_dir/'classification_report.csv', encoding='utf-8-sig')
    cm = confusion_matrix(y_test, predicted, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).rename_axis('true_stage').to_csv(
        output_dir/'confusion_matrix.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'SVM + FD + GLCM, seed {seed}')
    fig.tight_layout()
    fig.savefig(output_dir/'confusion_matrix.png', dpi=200)
    plt.close(fig)
    print(text)
    return metrics
