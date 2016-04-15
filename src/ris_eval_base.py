from __future__ import division

import cslab_environ

import argparse
import cv2
import numpy as np
import os
import tensorflow as tf

from data_api import cvppp
from data_api import kitti

from utils import logger
from utils.batch_iter import BatchIterator

import hungarian

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
            dataset['test']['label_segmentation'] = np.tile(
                dataset['test']['label_segmentation'], [1, 21, 1, 1])
            log.fatal(dataset['test']['label_score'].shape)
            dataset['test']['label_score'] = np.tile(
                dataset['test']['label_score'], [1, 21])
    elif dataset_name == 'kitti':
        dataset_folder = '/ais/gobi3/u/mren/data/kitti/object'
        dataset['train'] = kitti.get_dataset(
            dataset_folder, opt, split='train')
        dataset['valid'] = kitti.get_dataset(
            dataset_folder, opt, split='valid')
        dataset['valid_man'] = kitti.get_dataset(
            dataset_folder, opt, split='valid_man')
        dataset['test_man'] = kitti.get_dataset(
            dataset_folder, opt, split='test_man')
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
        idx_bat = dataset['index_map'][idx]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)
        return x_bat, y_bat, s_bat, idx_bat
    return get_batch


###############################
# Analysis helper functions
###############################
def _f_iou(a, b):
    """IOU between two segmentations.

    Args:
        a: [..., H, W], binary mask
        b: [..., H, W], binary mask

    Returns:
        dice: [...]
    """
    inter = (a * b).sum(axis=-1).sum(axis=-1)
    union = (a + b).sum(axis=-1).sum(axis=-1) - inter
    return inter / (union + 1e-5)


def _f_dice(a, b):
    """DICE between two segmentations.

    Args:
        a: [..., H, W], binary mask
        b: [..., H, W], binary mask

    Returns:
        dice: [...]
    """
    card_a = a.sum(axis=-1).sum(axis=-1)
    card_b = b.sum(axis=-1).sum(axis=-1)
    card_ab = (a * b).sum(axis=-1).sum(axis=-1)
    dice = 2 * card_ab / (card_a + card_b + 1e-5)
    return dice


def _f_best_dice(a, b):
    """For each a, look for the best DICE of all b.

    Args:
        a: [B, T, H, W], binary mask
        b: [B, T, H, W], binary mask

    Returns:
        best_dice: [B, T]
    """
    bd = np.zeros([a.shape[0], a.shape[1]])
    for ii in xrange(a.shape[1]):
        a_ = a[:, ii: ii + 1, :, :]
        dice = _f_dice(a_, b)
        bd[:, ii] = dice.max(axis=1)
        pass
    return bd


def _f_match(iou_pairwise):
    sess = tf.Session()
    tf_match = tf.user_ops.hungarian(
        tf.constant(iou_pairwise.astype('float32')))[0]
    return tf_match.eval(session=sess)


def f_ins_iou(y_out, y_gt, s_out, s_gt):
    """Calculates average instance-level IOU..

    Args:
        a: [B, T, H, W], binary mask
        b: [B, T, H, W], binary mask

    Returns:
        iou: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    y_out_ = np.expand_dims(y_out, 2)
    y_gt_ = np.expand_dims(y_gt, 1)
    iou_pairwise = _f_iou(y_out_, y_gt_)
    iou_pairwise = np.maximum(1e-4, iou_pairwise)
    iou_pairwise = np.round(iou_pairwise * 1e4) / 1e4
    match = _f_match(iou_pairwise)
    for ii in xrange(y_out.shape[0]):
        match[:, num_obj[ii]:, :] = 0.0
        match[:, :, num_obj[ii]:] = 0.0
    return (iou_pairwise * match).sum(axis=-1).sum(axis=-2) / num_obj


def f_symmetric_best_dice(y_out, y_gt, s_out, s_gt):
    """Calculates symmetric best DICE. min(BestDICE(a, b), BestDICE(b, a)).

    Args:
        a: [B, T, H, W], binary mask
        b: [B, T, H, W], binary mask

    Returns:
        sbd: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    bd1 = _f_best_dice(y_out, y_gt)
    bd1_mean = np.zeros([y_out.shape[0]])
    for ii in xrange(y_out.shape[0]):
        bd1_mean[ii] = bd1[ii, :num_obj[ii]].mean()
        pass
    bd2 = _f_best_dice(y_gt, y_out).mean()
    bd2_mean = np.zeros([y_out.shape[0]])
    for ii in xrange(y_out.shape[0]):
        bd2_mean[ii] = bd1[ii, :num_obj[ii]].mean()
        pass
    return np.minimum(bd1_mean, bd2_mean)


def f_coverage(y_out, y_gt, s_out, s_gt, weighted=False):
    """Calculates coverage score.

    Args:
        a: [B, T, H, W], binary mask
        b: [B, T, H, W], binary mask

    Returns:
        cov: [B]
    """
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

    return cov_mean


def f_wt_coverage(y_out, y_gt, s_out, s_gt):
    """Calculates weighted coverage score."""
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=True)


def f_unwt_coverage(y_out, y_gt, s_out, s_gt):
    """Calculates unweighted coverage score."""
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=False)


def f_fg_iou(y_out, y_gt, s_out, s_gt):
    """Calculates foreground IOU score."""
    return _f_iou(y_out.max(axis=1), y_gt.max(axis=1))


def f_fg_dice(y_out, y_gt, s_out, s_gt):
    """Calculates foreground DICE score."""
    return _f_dice(y_out.max(axis=1), y_gt.max(axis=1))


def f_count_acc(y_out, y_gt, s_out, s_gt):
    """Calculates count accuracy."""
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out == count_gt).astype('float')


def f_dic(y_out, y_gt, s_out, s_gt):
    """Calculates difference in count."""
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out - count_gt)


def f_dic_abs(y_out, y_gt, s_out, s_gt):
    """Calculates absolute difference in count."""
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return np.abs(count_out - count_gt)


def _f_count(s_out, s_gt):
    """Convert to count."""
    count_out = s_out.sum(axis=1)
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
        _tmp = self.func(y_out, y_gt, s_out, s_gt).sum()
        _num = y_out.shape[0]
        self.num_ex += _num
        self.avg += _tmp
        pass

    def finalize(self):
        self.avg /= self.num_ex
        log.info('{:20s}{:.4f}'.format(self.name, self.avg))
        pass


def run_eval(sess, m, dataset, batch_size=10, fname=None):
    analyzers = [StageAnalyzer('IOU', f_ins_iou, fname=fname),
                 StageAnalyzer('SBD', f_symmetric_best_dice, fname=fname),
                 StageAnalyzer('WT COV', f_wt_coverage, fname=fname),
                 StageAnalyzer('UNWT COV', f_unwt_coverage, fname=fname),
                 StageAnalyzer('FG DICE', f_fg_dice, fname=fname),
                 StageAnalyzer('FG IOU', f_fg_iou, fname=fname),
                 StageAnalyzer('COUNT ACC', f_count_acc, fname=fname),
                 StageAnalyzer('DIC', f_dic, fname=fname),
                 StageAnalyzer('|DIC|', f_dic_abs, fname=fname)]

    num_ex = dataset['input'].shape[0]
    batch_size = 10
    batch_iter = BatchIterator(num_ex,
                               batch_size=batch_size,
                               get_fn=get_batch_fn(dataset),
                               cycle=False,
                               progress_bar=True)
    _run_eval(sess, m, batch_iter, analyzers)

    # y_gt = [(np.random.rand(5, 5, 10, 10) > 0.5).astype('float')]
    # x = [None]
    # s_gt = [(np.random.rand(5, 5) > 0.5).astype('float')]
    # idx = [None]
    # batch_iter = zip(x, y_gt, s_gt, idx)
    # _run_eval(sess, m, batch_iter, analyzers)


def _run_eval(sess, m, batch_iter, analyzers):
    output_list = [m['y_out'], m['s_out']]
    for x, y_gt, s_gt, idx in batch_iter:
        feed_dict = {m['x']: x, m['y_gt']: y_gt, m['phase_train']: False}
        r = sess.run(output_list, feed_dict)
        y_out, s_out = postprocess(r[0], r[1])

        # y_out = (np.random.rand(5, 5, 10, 10) > 0.5).astype('float')
        # s_out = (np.random.rand(5, 5) > 0.5).astype('float')

        # y_gt = dataset.get_label(idx)
        # y_out = upsample(y_out, y_gt)

        [analyzer.stage(y_out, y_gt, s_out, s_gt) for analyzer in analyzers]
        pass
    [analyzer.finalize() for analyzer in analyzers]
    pass
