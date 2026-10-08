# ADMM-NeuralNetwork

Experimental centralized ADMM classifier: dense ReLU layers, one-hot binary hinge
loss, alternating least-squares and Cython output updates. No bias terms.

## Build and install

Requires Python 3.11+, NumPy 2+, SciPy 1.14+, scikit-learn 1.5+, and a C compiler.
Run commands from the repository directory.

Use the existing environment:

```sh
source .venv/bin/activate
python -m pip install -e '.[test,plot]'
```

If `.venv` does not exist, create it first with `python -m venv .venv`.
The install command builds Cython extensions and installs runtime dependencies,
pytest (`test` extra), and Matplotlib (`plot` extra). Build dependencies and NumPy
headers install automatically. Editable mode (`-e`) picks up Python source edits
without reinstalling. After editing `.pyx` files, rebuild:

```sh
python setup.py build_ext --inplace
```

## Test

```sh
python -m pytest -q
```

## Run benchmarks

```sh
admm-runner iris --repetitions 2 --iterations 20 --seed 42
admm-runner digits --repetitions 2 --iterations 20 --seed 42
```

`--repetitions` sets independent model runs; `--iterations` caps ADMM updates per
run, excluding warm-start steps. Both default to 10 and 100 respectively.
`--seed` defaults to 42 and controls splits and model initialization.
Benchmarks print mean held-out test accuracy, training time, and selected update
count. Use `admm-runner --help` for available options.

You can also use `python admm-runner.py` in place of `admm-runner` in every command.

## Plot

No Python script needed. Show a learning curve in a window:

```sh
admm-runner iris --plot curve --repetitions 2 --iterations 20 --seed 42
```

Save a plot without opening a window (also works on servers without a display):

```sh
admm-runner digits --plot curve --repetitions 2 --iterations 20 --output plots/digits.png
admm-runner iris --plot histogram --repetitions 20 --iterations 30 --output plots/iris-histogram.svg
```

`--output` requires `--plot`; parent directories are created automatically.
File extension selects format, including PNG, PDF, and SVG. Existing output files
are replaced. Omit `--output` to display either plot interactively.

- `--plot curve`: average training and validation accuracy over measured training
  time. Includes warm-start and all requested update points; no early stopping.
  Each repetition uses a seeded dataset split. Test labels are not inspected.
- `--plot histogram`: held-out test accuracy across independent initializations
  on the same seeded dataset split. Uses validation-based early stopping, just
  like the benchmark. More repetitions give a more useful distribution.

Plotting requires the `plot` extra, included in the installation command above.
Python users can still call `iris_fitting()` / `digits_fitting()` and histogram
helpers directly; these return Matplotlib figures for customization or saving.

## Data and evaluation

Arrays use `(features, samples)` and targets use `(classes, samples)` with exactly
one `1` per target column. Dimensions, finite values, binary targets, and positive
penalties are checked. `NeuralNetwork(..., rng=42)` reproduces initialization;
an existing `numpy.random.Generator` is also accepted.

Benchmarks use stratified train/test splits, then reserve 20% of training data for
validation. Validation alone selects a checkpoint (up to four stale updates or
99% validation accuracy); held-out test data is evaluated once per run afterward.
Reported time includes all attempted training updates; selected update count
excludes warm-start steps. Times may vary between runs; numerical results may
vary between BLAS/platforms.

Old benchmark scores/curves are not comparable: they selected iterations using
test accuracy and included synthetic curve points.
