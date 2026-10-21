# Qwen3.5-4B

The public data contain only 10 format examples. For formal training or evaluation, first set `AGS_DATA_DIR` to the complete local dataset directory; see the repository-root README.

Following Text S4, fixed-parameter nine-shot inference evaluates five configurations separately and reports the mean and sample standard deviation.
The full English prompts, pixel budget of 50176 to 200704, greedy decoding, maximum of 24 tokens, and disabled thinking follow the submitted materials.

```powershell
python src/download_model.py
python run_five.py
```

Run from this module directory. Model downloading requires network access; weights are not included in the repository.
The five nine-shot configurations are stored in `examples/config_01.json` through `examples/config_05.json`.
Each query image is excluded from the examples and, when necessary, replaced with the first other image from the same class and reactor. The filenames of the nine examples actually used are saved.
Aggregation requires successful inference and valid labels for all 231 records; configurations that have not run do not generate metrics.

Qwen uses a separate dependency environment: `pip install -r requirements.txt`. transformers is pinned to an exact Git commit based on the installed package metadata.
