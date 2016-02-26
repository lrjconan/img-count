import cslab_environ

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def random_flip_left_right(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(array_ops.pack([1.0, 1.0, uniform_random, 1.0]), 0.5)
    return array_ops.reverse(image, mirror)

def random_flip_up_down(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(array_ops.pack([1.0, uniform_random, 1.0, 1.0]), 0.5)
    return array_ops.reverse(image, mirror)
