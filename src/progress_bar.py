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

def get(length):
    return ProgressBar(length)

class ProgressBar(object):

    def __init__(self, length):
        self.length = length
        self.value = 0
        self.progress = 0
        self.width = 80
        self.finished = False
        pass

    def increment(self):
        self.value += 1
        while self.value / self.length > self.progress / self.width:
            sys.stdout.write('.')
            sys.stdout.flush()
            self.progress = self.progress + 1
        if self.progress == self.width and not self.finished:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.finished = True
        pass
