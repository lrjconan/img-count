from __future__ import division

import cslab_environ

import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from data_api import cvppp

from utils import logger
from utils.batch_iter import BatchIterator

log = logger.get()


def get_dataset(dataset_name, opt):
    if dataset_name == 'cvppp':
        if os.path.exists('/u/mren'):
            train_dataset_folder = '/ais/gobi3/u/mren/data/lsc/A1'
            test_dataset_folder = '/ais/gobi3/u/mren/data/lsc_test/A1'
            dataset = {}
            dataset['train'] = cvppp.get_dataset(
                train_dataset_folder, opt, split='train')
            dataset['valid'] = cvppp.get_dataset(
                train_dataset_folder, opt, split='valid')
            dataset['test'] = cvppp.get_dataset(
                test_dataset_folder, opt, split=None)
    else:
        raise Exception('Not supported')

    return dataset


def preprocess(x, y, s):
    return (x.astype('float32') / 255, y.astype('float32'), s.astype('float32'))


def postprocess(y_out, s_out):
    """Convert soft prediction to hard prediction."""
    s_mask = np.reshape(s_out, [-1, s_out.shape[1], 1, 1])
    y_out = y_out * s_mask
    y_out_max = np.argmax(y_out, axis=1)
    y_out_hard = np.zeros(y_out.shape)
    for idx in xrange(y_out.shape[1]):
        y_out_hard[:, idx] = (y_out_max == idx).astype(
            'float') * (y_out[:, idx] > 0.5).astype('float')
    s_out_hard = (s_out > 0.5).astype('float')
    return y_out_hard, s_out_hard


def get_batch_fn(dataset):
    """Preprocess mini-batch data given start and end indices."""
    def get_batch(idx):
        x_bat = dataset['input'][idx]
        y_bat = dataset['label_segmentation'][idx]
        s_bat = dataset['label_score'][idx]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)
        return x_bat, y_bat, s_bat
    return get_batch


###############################
# Analysis helper functions
###############################
def _f_iou(a, b):
    inter = (a * b).sum(axis=3).sum(axis=2)
    union = (a + b).sum(axis=3).sum(axis=2) - inter
    return inter / (union + 1e-5)


def _f_dice(a, b):
    card_a = a.sum(axis=3).sum(axis=2)
    card_b = b.sum(axis=3).sum(axis=2)
    card_ab = (a * b).sum(axis=3).sum(axis=2)
    dice = 2 * card_ab / (card_a + card_b + 1e-5)
    return dice


def _f_best_dice(a, b, num_obj):
    bd = np.zeros([a.shape[0], a.shape[1]])
    for ii in xrange(a.shape[1]):
        a_ = a[:, ii: ii + 1, :, :]
        dice = _f_dice(a_, b)
        bd[:, ii] = dice.max(axis=1)
        pass
    bd_mean = np.zeros([a.shape[0]])
    for ii in xrange(a.shape[0]):
        bd_mean[ii] = bd[ii, :num_obj[ii]].mean()
        pass
    return bd_mean


def f_symmetric_best_dice(y_out, y_gt, s_out, s_gt):
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    bd1 = _f_best_dice(y_out, y_gt, num_obj).mean()
    bd2 = _f_best_dice(y_gt, y_out, num_obj).mean()
    return min(bd1, bd2)


def f_coverage(y_out, y_gt, s_out, s_gt, weighted=False):
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    cov = np.zeros([y_out.shape[0], y_out.shape[1]])
    for ii in xrange(y_gt.shape[1]):
        y_gt_ = y_gt[:, ii: ii + 1, :, :]
        iou_ii = _f_iou(y_out, y_gt_)
        cov[:, ii] = iou_ii.max(axis=1)
        pass

    if weighted:
        weights = y_gt.sum(axis=3).sum(axis=2) / \
            y_gt.sum(axis=3).sum(axis=2).sum(axis=1, keepdims=True)
        pass
    else:
        weights = np.reshape(1 / num_obj, [-1, 1])
        pass

    cov *= weights
    cov_mean = np.zeros([y_out.shape[0]])
    for ii in xrange(y_out.shape[0]):
        cov_mean[ii] = cov[ii, :num_obj[ii]].sum()
        pass

    return cov_mean.mean()


def f_wt_coverage(y_out, y_gt, s_out, s_gt):
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=True)


def f_unwt_coverage(y_out, y_gt, s_out, s_gt):
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=False)


def f_fg_iou(y_out, y_gt, s_out, s_gt):
    return _f_iou(y_out.sum(axis=1), y_gt.sum(axis=1))


def f_fg_dice(y_out, y_gt, s_out, s_gt):
    return _f_dice(y_out.sum(axis=1), y_gt.sum(axis=1))


def f_count_acc(y_out, y_gt, s_out, s_gt):
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out == count_gt).astype('float').mean()


def f_dic(y_out, y_gt, s_out, s_gt):
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out - count_gt).mean()


def f_dic_abs(y_out, y_gt, s_out, s_gt):
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return np.abs(count_out - count_gt).mean()


def _f_count(s_out, s_gt):
    count_out = .sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    num_obj = np.maximum(count_gt, 1)
    return count_out, count_gt, num_obj


class StageAnalyzer(object):

    def __init__(self, name, func, fname=None):
        self.avg = 0.0
        self.num_ex = 0
        self.name = name
        self.func = func

    def stage(self, y_out, y_gt, s_out, s_gt):
        _tmp = self.func(y_out, y_gt, s_out, s_gt)
        _num = y_out.shape[0]
        self.num_ex += _num
        self.avg += _tmp * _num
        pass

    def finalize():
        self.avg /= self.num_ex
        log.info('{:20s}{:.4f}'.format(self.name, self.avg))
        pass


def build_matching():
    pass


def run_eval(sess, m, dataset, phase_train, batch_size=10, fname=None):
    analyzers = [StageAnalyzer('SBD', f_symmetric_best_dice, fname=fname),
                 StageAnalyzer('FG DICE', f_fg_dice, fname=fname),
                 StageAnalyzer('FG IOU', f_fg_iou, fname=fname),
                 StageAnalyzer('WT COV', f_wt_coverage, fname=fname),
                 StageAnalyzer('UNWT COV', f_unwt_coverage, fname=fname),
                 StageAnalyzer('COUNT ACC', f_count_acc, fname=fname),
                 StageAnalyzer('DIC', f_dic, fname=fname),
                 StageAnalyzer('|DIC|', f_dic_abs, fname=fname)]

    output_list = [m['y_out'], m['s_out']]
    num_ex = dataset['input'].shape[0]
    batch_size = 10
    batch_iter = BatchIterator(num_ex,
                               batch_size=batch_size,
                               get_fn=get_batch_fn(dataset),
                               cycle=False,
                               progress_bar=True)
    _run_eval(sess, output_list, batch_iter, phase_train, analyzers)


def _run_eval(sess, output_list, batch_iter, phase_train, analyzers):
    for x, y_gt, s_gt in batch_iter:
        feed_dict = {m['x']: x, m['y_gt']: y_gt, m['phase_train']: phase_train}
        r = sess.run(output_list, feed_dict)
        y_out, s_out = postprocess(r[0], r[1])
        [analyzer.stage(y_out, y_gt, s_out, s_gt) for analyzer in analyzers]
        pass
    [analyzer.finalize() for analyzer in analyzers]
    pass
