# Troubleshooting

Run this first whenever setup or data loading fails:

```bash
python scripts/diagnose.py --config configs/demo.yaml
```

Add `--require-gpu` on Colab when GPU execution is expected.

## Installation and runtime

### `ModuleNotFoundError: No module named 'fedgagan'`

Cause: the project is not installed, or the current directory is wrong.

```bash
cd /path/to/FedGAGAN-Reproduction
python -m pip install -e . --no-deps
python -m fedgagan --help
```

### `ModuleNotFoundError: No module named 'tensorflow'`

Local machine:

```bash
python -m pip install "tensorflow>=2.15,<3"
```

On Colab, TensorFlow is normally preinstalled. Restart the session if a prior
installation changed core packages, then verify:

```python
import tensorflow as tf
print(tf.__version__)
```

### TensorFlow/Keras dependency conflicts

Do not force-install another TensorFlow build on Colab unless necessary. Use:

```bash
python -m pip install -e . --no-deps
python -m pip install joblib matplotlib numpy pandas pyyaml scikit-learn seaborn
```

Restart the session after changing TensorFlow, Keras, NumPy, or protobuf.

### GPU list is empty

1. Select a GPU runtime.
2. Restart the Colab session.
3. Run `!nvidia-smi`.
4. Run `tf.config.list_physical_devices('GPU')`.

If `nvidia-smi` works but TensorFlow sees no GPU, the TensorFlow build was
probably replaced. Restart with a fresh runtime and avoid reinstalling it.

### `CUDA_ERROR_OUT_OF_MEMORY` or process killed

Reduce, in this order:

1. `train.batch_size`;
2. generator and critic hidden widths;
3. `ga.population_size` and `ga.evaluation_steps`;
4. `data.max_rows` for a feasibility run;
5. `evaluation.synthetic_samples`.

Start a fresh runtime after an OOM because GPU memory can remain fragmented.

## Dataset and preprocessing

### `No CSV files matched`

The glob in `data.paths` is wrong relative to the YAML file's project root.
Use an absolute path on Colab if necessary and verify:

```python
from glob import glob
print(glob('/content/drive/MyDrive/FedGAGAN/source/data/raw/UNSW-NB15/*.csv'))
```

### `Label column ... missing`

CSV headers are case-sensitive after surrounding spaces are stripped. Inspect:

```python
import pandas as pd
print(pd.read_csv('your.csv', nrows=3).columns.tolist())
```

Set `data.label_column` to the exact header.

### `Binary labels were not recovered`

`attack_values` did not produce both benign and attack classes. Inspect raw
labels:

```python
import pandas as pd
df = pd.read_csv('your.csv', low_memory=False)
print(df['Label'].value_counts(dropna=False).head(30))
```

List every attack label under `attack_values`; values not listed become benign.
Do not include the benign label.

### Too few attack rows per client

Each client needs at least two attack rows. Reduce `federated.num_clients`,
increase the dataset size, or correct the label mapping. Never duplicate rows
merely to satisfy the check because that distorts privacy and quality metrics.

### Preprocessing uses too much RAM

One-hot encoding a high-cardinality field such as IP address, flow ID, topic,
or timestamp can create thousands of columns. Add identifiers to
`data.drop_columns`, use `data.max_rows` while debugging, or engineer compact
features before this pipeline.

### `Input contains NaN`, infinity, or extremely large values

The loader replaces infinity with missing values and imputes columns, but a
column containing only missing values is unusable. Remove it or repair the raw
data. Inspect numeric ranges before training.

## WGAN-GP training

### Loss becomes `NaN` or `Inf`

Try:

- lower `learning_rate`, for example from `1e-4` to `5e-5`;
- verify all encoded values are finite;
- reduce batch size;
- keep `gradient_penalty_weight` near 10 initially;
- reset to `beta_1: 0.5`, `beta_2: 0.9`;
- use fewer/lower-width layers for a diagnostic run.

Discard a model trained after non-finite losses appear.

### Critic loss is large or oscillating

Wasserstein losses do not have the same interpretation as classification loss.
Oscillation alone is not proof of failure. Check generated-data KL,
mean/standard-deviation differences, correlations, PCA, and downstream F1.

### Generator produces nearly identical rows

Possible mode collapse. Lower the learning rate, increase training data, adjust
`n_critic`, verify minority samples are diverse, and compare several random
seeds. A low training loss alone does not establish synthetic-data quality.

### Generated categorical columns are fractional

The generator works in continuous encoded space. One-hot outputs are not
projected back to strict categories in this reproduction. Use the encoded CSV
for classifier experiments. If raw human-readable records are required, add an
argmax projection per one-hot group and validate its effect separately.

## GA and federated learning

### GA appears frozen

Paper-scale search is extremely expensive. Confirm GPU utilization and first
use 4-12 individuals, 2-5 generations, and 20-100 evaluation steps. Increase
only after timing one generation.

### Colab disconnects during GA

The current GA stage saves its final result after completion, not after every
candidate. Use a reduced search budget, keep the project/output directory in
Drive, and run stages separately. A disconnect during an unfinished `search`
stage requires restarting that stage.

### Federated training is slower than centralized training

This implementation simulates clients sequentially in one process. It models
the data isolation and FedAvg algorithm, not parallel distributed speedup.

### Privacy misconception

Raw rows stay inside each simulated client partition, but plain model updates
can still leak information. Do not claim formal privacy without secure
aggregation, differential privacy, and a separate privacy evaluation.

## Evaluation

### ROC AUC or precision is unexpectedly low

Check class counts, label mapping, leakage, and whether the real test set uses
the same feature schema. Compare against the real-only baseline before judging
augmentation. Synthetic data is useful only when held-out real performance
improves consistently across several seeds.

### Results differ from the paper

Exact numeric replication is not guaranteed because the paper omits exact raw
files/splits, all layer widths, client count, FL rounds, local epochs, client
selection, and partition details. Report this as a method reproduction and
document every chosen value.

### Plot generation fails in a headless environment

Set a non-interactive backend before running:

```bash
export MPLBACKEND=Agg
```

## Recovery checklist

1. Preserve `artifacts/` before resetting the runtime.
2. Rerun `scripts/diagnose.py` after reinstalling dependencies.
3. Do not rerun completed stages unless their inputs/configuration changed.
4. Delete an incompatible stage output only after copying it elsewhere.
5. Record the config, dataset checksum, seed, TensorFlow version, and GPU model
   for every experiment included in a report.

