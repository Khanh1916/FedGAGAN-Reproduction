# Google Colab Guide

This guide runs the repository in Google Colab without silently replacing
Colab's preinstalled TensorFlow build.

## 1. Select a GPU runtime

In Colab, select **Runtime > Change runtime type > T4 GPU**. GPU is strongly
recommended for WGAN-GP but the demo can run on CPU.

## 2. Keep work in Google Drive

Colab's local disk disappears when the runtime resets. Mount Drive and place
the ZIP plus all outputs there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create a project directory:

```python
from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/FedGAGAN')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
print(PROJECT_ROOT)
```

Upload `FedGAGAN-Reproduction.zip` into that directory using the Drive user
interface. Then extract it:

```python
import shutil
from pathlib import Path

zip_path = PROJECT_ROOT / 'FedGAGAN-Reproduction.zip'
source_root = PROJECT_ROOT / 'source'
if source_root.exists():
    print('Source already extracted:', source_root)
else:
    shutil.unpack_archive(zip_path, source_root)
%cd /content/drive/MyDrive/FedGAGAN/source
```

Do not repeatedly unzip over a modified project. Delete or rename `source`
first only when you intentionally want a fresh copy.

## 3. Install without downgrading Colab TensorFlow

First inspect the runtime:

```python
import sys
import tensorflow as tf

print('Python:', sys.version)
print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
```

Install the repository while preserving the TensorFlow already supplied by
Colab:

```python
!python -m pip install -q -e . --no-deps
!python -m pip install -q joblib matplotlib numpy pandas pyyaml scikit-learn seaborn
```

If Colab reports that packages were changed and asks for a restart, select
**Runtime > Restart session**, return to the project directory, and continue.

## 4. Run the preflight check

```python
!python scripts/diagnose.py --config configs/demo.yaml --require-gpu
```

The first run will report missing demo data. Create it, then rerun the check:

```python
!python -m fedgagan make-demo-data --output data/raw/demo.csv --rows 2400
!python scripts/diagnose.py --config configs/demo.yaml --require-gpu
```

Do not start model training until the final JSON line reports `"status":
"PASS"`.

## 5. Run the demo pipeline

```python
!python -m fedgagan run \
  --config configs/demo.yaml \
  --stages preprocess,train,generate,evaluate
```

Inspect the results:

```python
from pathlib import Path
import pandas as pd

output = Path('artifacts/demo')
display(pd.read_csv(output / 'classifier_metrics.csv'))
print((output / 'distribution_metrics.json').read_text())
```

Display generated plots:

```python
from IPython.display import Image, display

display(Image(filename='artifacts/demo/plots/means_and_stds.png'))
display(Image(filename='artifacts/demo/plots/pca_real_vs_synthetic.png'))
```

## 6. Add a real dataset

Store datasets under Drive so they survive a reset. For example:

```text
/content/drive/MyDrive/FedGAGAN/source/data/raw/UNSW-NB15/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

Copy `configs/paper.yaml` to a new configuration. Confirm:

- `data.paths` matches the uploaded files;
- `data.label_column` exactly matches the CSV header;
- `data.attack_values` contains every raw attack label;
- `data.drop_columns` does not remove a required feature;
- `output_dir` points to a Drive-backed directory.

Run only preprocessing first:

```python
!python scripts/diagnose.py --config configs/my_run.yaml --require-gpu
!python -m fedgagan run --config configs/my_run.yaml --stages preprocess
```

Check `data_summary.json` before spending GPU time.

## 7. Use a feasible Colab profile

The full paper GA requires about 5,000 candidate evaluations. Begin with:

```yaml
train:
  epochs: 10
  steps_per_epoch: 50
federated:
  num_clients: 3
  rounds: 3
  local_epochs: 1
ga:
  enabled: true
  population_size: 6
  generations: 3
  evaluation_steps: 30
evaluation:
  synthetic_samples: 2000
```

Run one stage at a time:

```python
!python -m fedgagan run --config configs/my_run.yaml --stages preprocess
!python -m fedgagan run --config configs/my_run.yaml --stages search
!python -m fedgagan run --config configs/my_run.yaml --stages train
!python -m fedgagan run --config configs/my_run.yaml --stages generate,evaluate
```

Completed stages write their results to `output_dir`. After a runtime reset,
remount Drive, reinstall the package, change to the source directory, and run
only the remaining stages.

## 8. Before a long run

1. Run the demo end to end.
2. Confirm GPU detection.
3. Inspect label counts and attack-row counts.
4. Keep `output_dir` in Google Drive.
5. Use a reduced GA first.
6. Keep the browser tab active during long free-tier sessions.
7. Download or copy critical output files after each stage.

See `TROUBLESHOOTING.md` for specific errors and recovery steps.

