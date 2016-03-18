"""
Foreground segmentation used to pre-train box proposal network.

Usage: python rec_ins_segm_attn.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time

from data_api import synth_shape
from data_api import cvppp
from data_api import kitti

from utils import log_manager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger
from utils import plot_utils

import fg_segm_models as models

if __name__ == '__main__':
    model_opt = {    
        'inp_height': 128,
        'inp_width': 448,
        'inp_depth': 3,
        'cnn_filter_size': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'cnn_depth': [4, 4, 8, 8, 16, 16, 32, 32, 64, 64]
        'cnn_pool': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        'dcnn_filter_size': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'dcnn_depth': [64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 1]
        'dcnn_pool': [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
        'use_bn': True,
        'rnd_hflip': True,
        'rnd_vflip': False,
        'rnd_transpose': False,
        'rnd_colour': False
    }
