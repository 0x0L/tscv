import numpy as np

class Sequencer(object):
    def __init__(self, n, lookback=1, lookforward=1, delay=0, step=1, squeeze=True):
        """Build sequence-liked features and targets indices for timeseries

            Idx / Time  0.......................................................n
            1           | lookback | delay | lookforward |                   |
            2           | step | lookback | delay | lookforward |            |
            ...
            last        | step | ... | step | lookback | delay | lookforward |

        Parameters
        ----------
        n : int
            Length of the timeseries.

        lookback : int (default is 1)
            Features are sequences built up taking `lookback` values in the past.

        lookforward : int (default is 1)
            Targets are sequences built up taking `lookforward` values in the
            future.

        delay : int (default is 0)
            Additional delay between features and targets.
            delay can be negative but must be greater than -lookback.

        step : int (default is 1)
            Stepping size between samples. Must be strictly positive.

        squeeze : boolean (default is True)
            If true, squeezes single timestep features and targets.
            Only has an effect if `lookback` or `lookforward` is equal to 1.

        Usage
        -----
        >>> X = np.random.randn(2500, 1000, 3)
        >>> y = X[:, :, 0]

        >>> seq = Sequencer(len(y), lookback=5, lookforward=1)
        >>> cv = seq.split(train_size=250, test_size=21)

        >>> for train_test_split in cv:
                X_train, X_test = seq.features(train_test_split, X)
                y_train, y_test = seq.targets(train_test_split, y)
                indices_train, indices_test = seq.indices(train_test_split)
        """
        squeeze_ = np.squeeze if squeeze else lambda x: x

        indices, features, targets = [], [], []
        for i in range(lookback, n + 1 - delay - lookforward, step):
            indices.append(i)
            features.append(squeeze_(
                    np.arange(i - lookback, i)))
            targets.append(squeeze_(
                    np.arange(i + delay, i + delay + lookforward)))

        self.indices_ = np.array(indices)
        self.features_, self.targets_ = np.array(features), np.array(targets)


    def split(self, train_size, test_size):
        """Vanilla rolling-window cross validation generator

        Parameters
        ----------
        train_size : int
            Size of the training set.

        test_size : int
            Size of the testing set.
        """
        n = len(self.indices_)
        for i in range(train_size, n, test_size):
            train, test = np.arange(i - train_size, i), np.arange(i, min(i + test_size, n))
            yield train, test


    def features(self, split, features):
        """Return features slice for a test/train split."""
        train, test = split
        return features[self.features_[train]], features[self.features_[test]]


    def targets(self, split, targets):
        """Return targets slice for a test/train split."""
        train, test = split
        return targets[self.targets_[train]], targets[self.targets_[test]]


    def indices(self, split):
        """Return indices slice for a test/train split."""
        train, test = split
        return self.indices_[train], self.indices_[test]
