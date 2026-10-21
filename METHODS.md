# Correspondence between paper methods and code

This file describes model configurations, data preprocessing, and statistical analysis methods. The repository provides 10 public examples; complete experiments use the complete local dataset.

## Model inputs and directories

| Paper model | Directory | Actual inputs |
|---|---|---|
| SVM | SVM + FD + GLCM texture features | FD, mean grayscale intensity, grayscale population standard deviation, and GLCM contrast/correlation/energy/homogeneity |
| UNM512 | UNM512 | FD、D50、Δt |
| EfficientNet-B0 | EfficientNet_B0 | RGB microscopic images |
| ViT-S/16 | ViT_S/16 | RGB microscopic images |
| ResNet18/34/50/101/152 | Directories with the same names | RGB microscopic images, D50, and Δt |
| ResNet18FD/34FD/50FD/101FD/152FD | Directories with the same names | RGB microscopic images, FD, D50, and Δt |
| ResNet34FD-Gated | ResNet34FDGated | RGB microscopic images, FD, D50, and Δt; gated fusion of the numerical branch |
| Qwen3.5-4B | Qwen3.5_4B_VL | Nine labeled examples, a query image, and morphological prompts; no numerical inputs |

The model architecture includes a numerical encoding branch, a ResNet backbone, spatial attention, a classification head, and gated fusion.

## Shared training methods

Each three-fold run holds out Z1 (76 images), Z2 (78 images), and Z3 (77 images) in turn. The other two reactors are randomly split 80:20, with the validation set rounded up to 31 images. The corresponding training set sizes are 124, 122, and 123. The defaults are shuffle=True and non-stratified sampling; the manuscript does not specify a stratification rule. All models use the same splitting function and seed rule.

The settings are batch size=12, 100 epochs, 5 repetitions, an initial learning rate of 0.0005, Adam, CrossEntropyLoss, and the original weight_decay=1e-4. The learning rate is multiplied by 0.1 after 10 consecutive epochs without improvement in validation loss. After all training epochs, the weights with the lowest independent validation loss are restored; test data are used only for final evaluation. The random seed is 42+100×(repetition index−1)+(fold index−1). The 20 paired runs use the same formula, with identical fold memberships across the three models.

Images are 224×224. ResNet/EfficientNet use ImageNet means and standard deviations; ViT uses bicubic interpolation and mean/std=0.5 for each channel. No random augmentation is used. The numerical branch uses float32 directly without standardization. SVM fits StandardScaler only on the training set. ViT retains the original CUDA mixed-precision implementation; gradient accumulation defaults to 1.

The 231 independent test predictions from all three folds are pooled within each complete run to calculate Accuracy and Macro-Precision/Recall/F1. The mean and sample standard deviation (ddof=1) are then calculated across repetitions. Macro-F1 is the arithmetic mean of the three class-specific F1 scores.

## FD preprocessing

The public source records are stored in data/example_data/data.csv and contain only 10 rows. Following Text S2, foreground pixels with grayscale values <= the Otsu threshold are counted across the whole image using a grid fixed at the upper-left corner and box sizes of 5/10/20/40/80/160/320 pixels. ln[N] is fitted against ln[L/ε] at the 20/40/80-pixel scales. By default, the examples are processed to generate data/example_data/data_paper_method.csv. AGS_DATA_DIR can instead select a complete local dataset for reading and processing. All models read data_paper_method.csv from the selected directory. Incomplete edge boxes are counted using the pixels they actually cover; edge image content is not removed.

## Qwen configuration

Five fixed example configurations store complete image filenames. Each contains three images per class, with one from each reactor. If a query image is an example, it is replaced with the first other sample from the same class and reactor. During evaluation on the complete dataset, the actual example lists are saved with the per-image inference results.

The full English system and query prompts follow Text S4. Settings are a pixel budget of 50176–200704, seed=42, thinking=False, evaluation/inference mode, greedy decoding, and at most 24 tokens. The five configurations are aggregated after each has actually completed inference; failed or invalid labels are excluded from the final five-configuration summary. Model weights must be downloaded separately.

## Analysis commands

Run from the repository root:

```powershell
python analysis/fd.py --recalculate
python analysis/deep_inputs.py --project ResNet34FD --mode permutation
python analysis/deep_inputs.py --project ResNet34FD --mode correlation
python analysis/paired_tests.py --resnet ResNet34/results_20 --fd ResNet34FD/results_20 --gated ResNet34FDGated/results_20
```

By default, deep_inputs requires weights and training split manifests for 5 repetitions × 3 independent leave-one-reactor-out folds per model; older weights with overlapping test samples are not accepted. Each input is permuted 30 times only within its independent test reactor. Accuracy differences are first calculated from the pooled 231 predictions across three folds, then summarized as the mean and sample standard deviation across training repetitions. FD correlations use the median |rho| across valid image channels, average the three folds first, and then summarize the 5 repetitions.

Paired tests use 20 complete three-fold runs, each with 231 predictions. Metrics are recalculated from per-image records after verifying common samples and identical splits. Zero differences are excluded, and tied absolute differences receive average ranks. Dynamic programming over observed ranks exactly counts all sign assignments. Floating-point differences are rounded to 12 decimal places. Holm correction is applied to the two comparisons separately for Macro-F1 and Accuracy.

## Runtime checks

Public mode checks the correspondence between 10 images and two 10-row CSV files, image checksums, the six-column input format, model parameters, and program syntax. The 10 examples do not support the complete paper experiment; training entry points prompt for the complete local dataset. `verify_repository.py --full` checks the complete dataset and independent splits.

