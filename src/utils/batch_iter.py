"""
A batch iterator.

Usage:
    for inp_batch, lb_batch in BatchIterator(inputs, labels, 25):
        train(inp_batch, lb_batch)
"""

import numpy as np


class BatchIterator(object):

    def __init__(self, data, labels=None, batch_size=1):
        """Construct a batch iterator.

        Args:
            data: numpy.ndarray, (N, D), N is the number of examples, D is the
            feature dimension.
            labels: numpy.ndarray, (N), N is the number of examples.
            batch_size: int, batch size.
        """

        self._batch_size = batch_size
        self._step = 0
        self._num_ex = data.shape[0]
        if labels is not None and self._num_ex != labels.shape[0]:
            raise Exception('Data and labels shape do not match.')
        self._num_steps = np.ceil(self._num_ex / batch_size)
        self._data = data
        self._labels = labels
        pass

    def __iter__(self):
        """Get iterable."""
        return self

    def __len__(self):
        """Get iterable length."""
        return self._num_steps

    def next(self):
        """Iterate next element."""
        if self._step < self._num_steps:
            start = self._batch_size * self._step
            end = min(self._num_ex, self._batch_size * (self._step + 1))
            self._step += 1
            if self._labels is not None:
                return self._data[start: end], self._labels[start: end]
            else:
                return self._data[start: end]
        else:
            raise StopIteration()

        pass
