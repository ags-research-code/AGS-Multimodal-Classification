
import gc
import os
import random
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_SEED, CSV_PATH, DEVICE, VALIDATION_FRACTION,
                    LEARNING_RATE, MODEL_SAVE_DIR, NUM_EPOCHS, NUM_REPEATS,
                    RESULTS_DIR, WEIGHT_DECAY)
from data_loader import get_data_loaders, read_labeled_csv
from model import MultiModalModel

FOLD_ORDER = ('Z1', 'Z2', 'Z3')


def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_confusion_matrix(y_true, y_pred, classes, output_dir, title):
    
    output_dir = Path(output_dir)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        output_dir / 'confusion_matrix.csv', encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    for (row, column), value in np.ndenumerate(cm):
        ax.text(column, row, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)


class Trainer:
    

    def __init__(self, repeat_id, fold_id, test_reactor, seed):
        self.repeat_id = repeat_id
        self.fold_id = fold_id
        self.test_reactor = test_reactor
        self.seed = seed
        set_random_seed(seed)

        folder_name = f'fold_{fold_id:02d}_test_{test_reactor}'
        self.results_dir = Path(RESULTS_DIR) / f'repeat_{repeat_id:02d}' / folder_name
        self.model_dir = Path(MODEL_SAVE_DIR) / f'repeat_{repeat_id:02d}' / folder_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        (self.train_loader, self.val_loader, self.test_loader,
         self.num_classes) = get_data_loaders(
            test_reactor=test_reactor,
            random_seed=seed,
            results_dir=self.results_dir,
        )
        self.label_encoder = self.train_loader.dataset.label_encoder
        self.classes = list(self.label_encoder.classes_)
        self.model = MultiModalModel(num_classes=self.num_classes).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
            'learning_rate': [],
        }

    def train_epoch(self):
        self.model.train()
        loss_sum, samples_seen = 0.0, 0
        predictions, targets = [], []
        for batch in tqdm(
                self.train_loader,
                desc=f'R{self.repeat_id} F{self.fold_id} Training'):
            images = batch['images'].to(DEVICE)
            data = batch['data'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            self.optimizer.zero_grad()
            logits = self.model(images, data)
            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * target.size(0)
            samples_seen += target.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(target.cpu().tolist())
        return loss_sum / samples_seen, accuracy_score(targets, predictions)

    def validate(self):
        self.model.eval()
        loss_sum, sample_count = 0.0, 0
        predictions, targets = [], []
        with torch.inference_mode():
            for batch in tqdm(
                    self.val_loader,
                    desc=f'R{self.repeat_id} F{self.fold_id} Validation {self.test_reactor}'):
                images = batch['images'].to(DEVICE)
                data = batch['data'].to(DEVICE)
                target = batch['target'].to(DEVICE)
                logits = self.model(images, data)
                loss_sum += self.criterion(logits, target).item() * target.size(0)
                sample_count += target.size(0)
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(target.cpu().tolist())
        val_accuracy = accuracy_score(targets, predictions)
        val_macro_f1 = f1_score(
            targets, predictions, labels=list(range(self.num_classes)),
            average='macro', zero_division=0)
        return loss_sum / sample_count, val_accuracy, val_macro_f1

    def evaluate_test_once(self):
        self.model.eval()
        loss_sum, sample_count = 0.0, 0
        actual, predicted = [], []
        image_names, reactor_ids, used_flags = [], [], []
        with torch.inference_mode():
            for batch in tqdm(
                    self.test_loader,
                    desc=f'R{self.repeat_id} F{self.fold_id} Final test {self.test_reactor}'):
                images = batch['images'].to(DEVICE)
                data = batch['data'].to(DEVICE)
                target = batch['target'].to(DEVICE)
                logits = self.model(images, data)
                loss_sum += self.criterion(logits, target).item() * target.size(0)
                sample_count += target.size(0)
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
                actual.extend(target.cpu().tolist())
                image_names.extend(batch['image_name'])
                reactor_ids.extend(batch['reactor_id'])
                used_flags.extend(batch['was_used_for_training'].cpu().tolist())

        y_true = self.label_encoder.inverse_transform(np.asarray(actual))
        y_pred = self.label_encoder.inverse_transform(np.asarray(predicted))
        predictions_df = pd.DataFrame({
            'repeat_id': self.repeat_id,
            'fold_id': self.fold_id,
            'fold_seed': self.seed,
            'test_reactor': self.test_reactor,
            'image': image_names,
            'reactor_id': reactor_ids,
            'was_used_for_training': used_flags,
            'true_stage': y_true,
            'predicted_stage': y_pred,
        })
        predictions_df['correct'] = (
            predictions_df['true_stage'] == predictions_df['predicted_stage'])
        predictions_df.to_csv(
            self.results_dir / 'final_test_predictions.csv',
            index=False, encoding='utf-8-sig')

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=self.classes,
            average='macro', zero_division=0)
        weighted_f1 = f1_score(
            y_true, y_pred, labels=self.classes,
            average='weighted', zero_division=0)
        report = classification_report(
            y_true, y_pred, labels=self.classes,
            target_names=self.classes, zero_division=0)
        summary = (
            f'Repeat: {self.repeat_id}\nFold: {self.fold_id}\n'
            f'Test reactor: {self.test_reactor}\nSeed: {self.seed}\n'
            f'Test samples: {sample_count}\n'
            f'Used for training: {int(predictions_df["was_used_for_training"].sum())}\n'
            f'Test loss: {loss_sum / sample_count:.6f}\n'
            f'Accuracy: {accuracy:.6f}\nMacro-F1: {macro_f1:.6f}\n'
            f'Weighted-F1: {weighted_f1:.6f}\n\n{report}')
        print('\n' + summary)
        (self.results_dir / 'final_test_report.txt').write_text(
            summary, encoding='utf-8')
        save_confusion_matrix(
            y_true, y_pred, self.classes, self.results_dir,
            f'Repeat {self.repeat_id}, fold {self.fold_id}: test {self.test_reactor}')
        return predictions_df, {
            'repeat_id': self.repeat_id,
            'fold_id': self.fold_id,
            'test_reactor': self.test_reactor,
            'seed': self.seed,
            'samples': sample_count,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
        }

    def train(self):
        print(
            f'\n=== Repeat {self.repeat_id}/{NUM_REPEATS}, '
            f'Fold {self.fold_id}/3, test {self.test_reactor}, seed {self.seed} ===')
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        for epoch in range(NUM_EPOCHS):
            train_start = perf_counter()
            train_loss, train_acc = self.train_epoch()
            train_seconds = perf_counter() - train_start
            val_start = perf_counter()
            val_loss, val_acc, val_macro_f1 = self.validate()
            val_seconds = perf_counter() - val_start
            self.scheduler.step(val_loss)
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_macro_f1'].append(val_macro_f1)
            self.history['learning_rate'].append(learning_rate)
            print(
                f'Epoch {epoch + 1}/{NUM_EPOCHS} - '
                f'Train Loss={train_loss:.4f}, Acc={train_acc:.4f}; '
                f'Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, '
                f'Macro-F1={val_macro_f1:.4f}; LR={learning_rate:.2g}; '
                f'Train={train_seconds:.1f}s, Val={val_seconds:.1f}s')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }

        if best_model_state is None:
            raise RuntimeError('No model parameters were available for saving during training.')
        self.model.load_state_dict(best_model_state)
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'classes': self.classes,
            'selection_method': 'lowest_independent_validation_loss',
            'best_val_loss': best_val_loss,
            'repeat_id': self.repeat_id,
            'fold_id': self.fold_id,
            'seed': self.seed,
            'test_reactor': self.test_reactor,
            'validation_fraction': VALIDATION_FRACTION,
            'protocol': 'loro_independent_80_20_v1',
        }
        # Save through a file object to avoid ASCII encoding of Chinese paths in some Windows/PyTorch versions.
        self.model_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = self.model_dir / 'final_model.pth'
        with final_model_path.open('wb') as model_file:
            torch.save(checkpoint, model_file)

        pd.DataFrame(self.history).to_csv(
            self.results_dir / 'training_history.csv', index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.history['train_loss'], label='Train loss')
        ax.plot(self.history['val_loss'], label=f'Validation loss ({self.test_reactor})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross entropy loss')
        ax.set_title(
            f'Repeat {self.repeat_id}, fold {self.fold_id}: {self.test_reactor}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.results_dir / 'training_loss.png', dpi=200)
        plt.close(fig)
        print(
            f'Restored the parameters with the lowest validation loss: epoch={best_epoch}, '
            f'val_loss={best_val_loss:.6f}')
        return self.evaluate_test_once()


def save_repeat_report(repeat_id, predictions, fold_metrics, classes, expected_images):
    
    output_dir = Path(RESULTS_DIR) / f'repeat_{repeat_id:02d}' / 'combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(predictions, ignore_index=True).sort_values(['reactor_id','image']).reset_index(drop=True)
    if len(combined)!=len(expected_images) or combined.image.duplicated().any() or set(combined.image)!=set(expected_images):
        raise ValueError('The three-fold predictions do not cover all independent test images exactly once.')
    if combined.was_used_for_training.any():
        raise ValueError('Independent test predictions contain samples used to train the corresponding fold.')
    y_true,y_pred=combined.true_stage,combined.predicted_stage
    metrics={
        'accuracy':accuracy_score(y_true,y_pred),
        'precision':precision_score(y_true,y_pred,labels=classes,average='macro',zero_division=0),
        'recall':recall_score(y_true,y_pred,labels=classes,average='macro',zero_division=0),
        'macro_f1':f1_score(y_true,y_pred,labels=classes,average='macro',zero_division=0),
    }
    combined.to_csv(output_dir/'combined_predictions_231.csv',index=False,encoding='utf-8-sig')
    summary=f'Repeat: {repeat_id}; Independent LORO; n={len(combined)}\n'
    summary+='\n'.join(f'{k}: {v:.8f}' for k,v in metrics.items())
    summary+='\n\n'+classification_report(y_true,y_pred,labels=classes,zero_division=0)
    (output_dir/'combined_report_231.txt').write_text(summary,encoding='utf-8')
    pd.DataFrame(fold_metrics).to_csv(output_dir/'fold_metrics.csv',index=False)
    save_confusion_matrix(y_true,y_pred,classes,output_dir,f'Repeat {repeat_id}: independent 3 folds')
    return dict(repeat_id=repeat_id,samples=len(combined),**metrics)


def save_five_repeat_summary(repeat_metrics):
    
    output_dir = Path(RESULTS_DIR) / ('five_repeats_summary' if NUM_REPEATS==5 else f'{NUM_REPEATS}_repeats_summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(repeat_metrics)
    metrics_df.to_csv(
        output_dir / 'repeat_metrics.csv', index=False, encoding='utf-8-sig')
    numeric_columns = [
        column for column in metrics_df.columns
        if column not in {'repeat_id', 'samples', 'seen_samples', 'unseen_samples'}
    ]
    statistics = pd.DataFrame({
        'mean': metrics_df[numeric_columns].mean(),
        'std': metrics_df[numeric_columns].std(ddof=1),
    })
    statistics.to_csv(
        output_dir / 'mean_and_std.csv', encoding='utf-8-sig')
    text = f'{NUM_REPEATS}-repeat metrics\n\n' + metrics_df.to_string(index=False)
    text += '\n\nMean and standard deviation\n\n' + statistics.to_string()
    (output_dir / 'summary.txt').write_text(text, encoding='utf-8')
    print('\n' + text)


def run_experiment():
    
    print(f'Model compute device DEVICE={DEVICE!s}; CUDA available={torch.cuda.is_available()}')
    if str(DEVICE).startswith('cuda') and torch.cuda.is_available():
        print(f'Current GPU: {torch.cuda.get_device_name(DEVICE)}')
    source_df = read_labeled_csv(CSV_PATH)
    expected_images = source_df['image'].tolist()
    classes = sorted(source_df['stage'].unique().tolist())
    print('Data distribution (rows=reactor, columns=stage):')
    print(pd.crosstab(source_df['reactor_id'], source_df['stage']).to_string())
    print(
        f'\nAbout to run {NUM_REPEATS} repetitions × 3 folds × {NUM_EPOCHS} epoch, '
        f'Training a total of {NUM_REPEATS * 3} independent models.')

    repeat_metrics = []
    for repeat_id in range(1, NUM_REPEATS + 1):
        predictions, fold_metrics = [], []
        for fold_id, test_reactor in enumerate(FOLD_ORDER, start=1):
            seed = BASE_SEED + (repeat_id - 1) * 100 + (fold_id - 1)
            trainer = Trainer(repeat_id, fold_id, test_reactor, seed)
            fold_predictions, metrics = trainer.train()
            predictions.append(fold_predictions)
            fold_metrics.append(metrics)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        repeat_metrics.append(save_repeat_report(
            repeat_id, predictions, fold_metrics, classes, expected_images))
    save_five_repeat_summary(repeat_metrics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Independent leave-one-reactor-out training; the paper defaults to 5 repetitions, with 20 available for paired tests')
    parser.add_argument('--num-repeats', type=int, default=NUM_REPEATS)
    parser.add_argument('--base-seed', type=int, default=BASE_SEED)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--model-dir', type=Path, default=MODEL_SAVE_DIR)
    args = parser.parse_args()
    if args.num_repeats < 1:
        parser.error('--num-repeats must be a positive integer')
    NUM_REPEATS, BASE_SEED = args.num_repeats, args.base_seed
    RESULTS_DIR, MODEL_SAVE_DIR = args.results_dir, args.model_dir
    run_experiment()
