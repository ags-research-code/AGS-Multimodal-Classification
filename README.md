# AGS granulation stage identification

This repository provides training and analysis programs for 16 model configurations. The publicly available data consist of only 10 real image examples and their corresponding CSV records, collectively called "example data". The complete experimental dataset is not included in this repository.

## Example data

Located in `data/example_data/`:

- 10 original microscopic images with unchanged resolution and image content.
- `data.csv`: the source-table records corresponding to the 10 selected images, with the original values unchanged.
- `data_paper_method.csv`: the FD input table calculated for the same 10 images following Text S2.
- `dataset_info.json`: example counts, class distribution, and image checksums.

The examples include 4 growth, 3 stabilization, and 3 disintegration images and cover reactors Z1, Z2, and Z3. They demonstrate the file format, image loading, and FD preprocessing. The limited example set does not support complete three-fold training or per-image nine-shot Qwen evaluation.

```powershell
pip install -r requirements.txt
python verify_repository.py
python prepare_data.py
```

Run the commands above from the repository root. By default, they only check or process the 10 examples without downloading weights or running complete training.

## Using a complete local dataset

Place the complete images and a headerless six-column `data.csv` in the same private directory. The CSV columns are image filename, FD, D50, Δt, stage, and reactor, in that order.

```powershell
$env:AGS_DATA_DIR = "C:/path/to/private_dataset"
python prepare_data.py
python verify_repository.py --full
python run_experiments.py --models ResNet34FD
```

`AGS_DATA_DIR` only changes the local input location; it does not copy private data into the repository. Complete experiments default to 5 repetitions of independent leave-one-reactor-out three-fold evaluation, with 100 epochs per fold. `python run_experiments.py --paired` runs 20 comparisons for the three specified models. Supervised image models require pretrained weights on their first run; weights and results are not distributed with the repository.

After configuring the complete local dataset, run Qwen using its module-specific `requirements.txt`, `src/download_model.py`, and `run_five.py`. The 10 public examples are insufficient for this evaluation.

## Methods

See `METHODS.md` for model inputs, optimization settings, interpretation analyses, and statistical entry points.

`.gitignore` excludes model weights, results, logs, and caches. Keep the complete private dataset outside the repository.
