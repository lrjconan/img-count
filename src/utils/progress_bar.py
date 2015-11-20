"""
A python progress bar

Usage:
    import progress_bar

    N = 1000
    pb = progress_bar.get(N)
    
    for i in xrange(N):
        pb.increment()
"""


from __future__ import division
import sys


def get(length, iterable=None):
    """Returns a ProgressBar object.

    Args:
        length: number, total number of objects to count.
    """
    return ProgressBar(length, iterable=iterable)


class ProgressBar(object):
    """Prints a dotted line in a standard terminal."""

    def __init__(self, length, iterable=None, width=80):
        """Constructs a ProgressBar object.

        Args:
            length: number, total number of objects to count.
            width: number, width of the progress bar.
        """
        self.length = length
        self.value = 0
        self.progress = 0
        self.iterable = iterable
        self.width = width
        self._finished = False
        pass

    def __iter__(self):
        """Get iterable object."""
        return self

    def next(self):
        """Iterate next."""
        if self.iterable:
            self.increment()
            return self.iterable.next()
        else:
            return self.value

    def increment(self, value=1):
        """Increments the progress bar.

        Args:
            value: number, value to be incremented, default 1.
        """
        self.value += value
        while self.value / self.length > self.progress / self.width:
            sys.stdout.write('.')
            sys.stdout.flush()
            self.progress = self.progress + 1
        if self.progress == self.width and not self._finished:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._finished = True
        pass
