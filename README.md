# tscv

Timeseries cross-validation tools for Python 3

## Intro

When dealing with time-distributed data in the context of neural networks, I always ended up writing some ad hoc code for the following tasks:
* build lagged matrix views of my data
* perform a rolling-window cross-validation

The `tscv` module combines these two things into a single object.

## Usage

Let's start by creating some arbitrary time-distributed data. The first dimension is the time dimension and it must agree accross all our data.

```python
N = 200  # number of timesteps

from numpy.random import randn
s = randn(N, 1000, 3)
u = randn(N, 23)
```

In this contrived example, let's assume our task is to predict the next 2 values (our `targets`) of our timeseries using the last 10 values (our `features`). The `Sequencer` class from `tscv` helps in building sequences spanning past and futures values of the timeseries we just created. In our example, we want

```python
from tscv import Sequencer
seq = Sequencer(N, lookback=10, lookforward=2)
```

To deal with possible non-stationarity in the data, we want to do a rolling-window cross-validation. In our case, let's say we want to learn on the past 50 samples and predict the next 10 ones. The `Sequencer` class has a method to create such cross-validation strategies

```python
cv = seq.split(train_size=50, test_size=10)
```

To access `features` or `targets` slices, we use the `features` and `targets` methods on the `Sequence` object. If `s` is the predictor and `u` is the predicted value

```python
for split in cv:
    X_train, X_test = seq.features(split, s)
    y_train, y_test = seq.targets(split, u)
    t_train, t_test = seq.indices(split)
```
