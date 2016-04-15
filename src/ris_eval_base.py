from __future__ import division

import cslab_environ

import argparse
import cv2
import numpy as np
import os
import tensorflow as tf

from data_api.cvppp import CVPPP
from data_api.kitti import KITTI

from utils import logger
from utils.batch_iter import BatchIterator

import hungarian

log = logger.get()


def get_dataset(dataset_name, opt):
    """Get dataset, including test."""
    if dataset_name == 'cvppp':
        if os.path.exists('/u/mren'):
            train_dataset_folder = '/ais/gobi3/u/mren/data/lsc/A1'
            test_dataset_folder = '/ais/gobi3/u/mren/data/lsc_test/A1'
            dataset = {}
            dataset['train'] = CVPPP(train_dataset_folder, opt, split='train')
            dataset['valid'] = CVPPP(train_dataset_folder, opt, split='valid')
            dataset['test'] = CVPPP(
                test_dataset_folder, opt, split=None, manual_max=21)
    elif dataset_name == 'kitti':
        dataset_folder = '/ais/gobi3/u/mren/data/kitti/object'
        dataset['train'] = KITTI(dataset_folder, opt, split='train')
        dataset['valid'] = KITTI(dataset_folder, opt, split='valid')
        dataset['valid_man'] = KITTI(dataset_folder, opt, split='valid_man')
        dataset['test_man'] = KITTI(dataset_folder, opt, split='test_man')
    else:
        raise Exception('Not supported')
    return dataset


def preprocess(x, y, s):
    """Preprocess input data."""
    return (x.astype('float32') / 255,
            y.astype('float32'), s.astype('float32'))


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
        a: [T, H, W], binary mask
        b: [T, H, W], binary mask

    Returns:
        best_dice: [T]
    """
    bd = np.zeros([a.shape[0]])
    for ii in xrange(a.shape[0]):
        a_ = a[ii: ii + 1, :, :]
        dice = _f_dice(a_, b)
        bd[ii] = dice.max(axis=0)
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
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        ins_iou: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    ins_iou = np.zeros([num_ex])
    for ii in xrange(num_ex):
        y_out_ = np.expand_dims(y_out[ii], 1)
        y_gt_ = np.expand_dims(y_gt[ii], 0)
        iou_pairwise = _f_iou(y_out_, y_gt_)
        iou_pairwise = np.maximum(1e-4, iou_pairwise)
        iou_pairwise = np.round(iou_pairwise * 1e4) / 1e4
        match = _f_match(iou_pairwise)
        match[num_obj[ii]:, :] = 0.0
        match[:, num_obj[ii]:] = 0.0
        ins_iou[ii] = (iou_pairwise * match).sum(
            axis=-1).sum(axis=-1) / num_obj[ii]
    return ins_iou


def f_symmetric_best_dice(y_out, y_gt, s_out, s_gt):
    """Calculates symmetric best DICE. min(BestDICE(a, b), BestDICE(b, a)).

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        sbd: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)

    def f_bd(a, b):
        num_ex = len(a)
        timespan = a[0].shape[0]
        bd = np.zeros([num_ex, timespan])
        for ii in xrange(num_ex):
            bd[ii] = _f_best_dice(a[ii], b[ii])
        bd_mean = np.zeros([a.shape[0]])
        for ii in xrange(a.shape[0]):
            bd_mean[ii] = bd[ii, :num_obj[ii]].mean()
        return bd
    return np.minimum(f_bd(y_out, y_gt), f_bd(y_gt, y_out))


def f_coverage(y_out, y_gt, s_out, s_gt, weighted=False):
    """Calculates coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    cov = np.zeros([num_ex, timespan])

    for ii in xrange(num_ex):
        for jj in xrange(timespan):
            # [1, H, W]
            y_gt_ = y_gt[ii][jj: jj + 1, :, :]
            iou_jj = _f_iou(y_out[ii], y_gt_)
            cov[ii, jj] = iou_jj.max(axis=0)
            pass

    weights = np.zeros([num_ex, timespan])
    for ii in xrange(num_ex):
        if weighted:
            weights[ii] = y_gt[ii].sum(axis=-1).sum(axis=-1) / \
                (y_gt[ii].sum() + 1e-5)
        else:
            weights[ii] = 1 / num_obj[ii]

    cov *= weights
    cov_mean = np.zeros([y_out.shape[0]])
    for ii in xrange(y_out.shape[0]):
        cov_mean[ii] = cov[ii, :num_obj[ii]].sum()
        pass

    return cov_mean


def f_wt_coverage(y_out, y_gt, s_out, s_gt):
    """Calculates weighted coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=True)


def f_unwt_coverage(y_out, y_gt, s_out, s_gt):
    """Calculates unweighted coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    return f_coverage(y_out, y_gt, s_out, s_gt, weighted=False)


def f_fg_iou(y_out, y_gt, s_out, s_gt):
    """Calculates foreground IOU score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        fg_iou: [B]
    """
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    fg_iou = np.zeros([num_ex])
    for ii in xrange(num_ex):
        fg_iou[ii] = _f_iou(y_out[ii].max(axis=0), y_gt[ii].max(axis=0))
    return fg_iou


def f_fg_dice(y_out, y_gt, s_out, s_gt):
    """Calculates foreground DICE score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        fg_dice: [B]
    """
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    fg_dice = np.zeros([num_ex])
    for ii in xrange(num_ex):
        fg_dice[ii] = _f_dice(y_out[ii].max(axis=0), y_gt[ii].max(axis=0))
    return fg_dice


def f_count_acc(y_out, y_gt, s_out, s_gt):
    """Calculates count accuracy.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        count_acc: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out == count_gt).astype('float')


def f_dic(y_out, y_gt, s_out, s_gt):
    """Calculates difference in count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        dic: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return (count_out - count_gt)


def f_dic_abs(y_out, y_gt, s_out, s_gt):
    """Calculates absolute difference in count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        dic_abs: [B]
    """
    count_out, count_gt, num_obj = _f_count(s_out, s_gt)
    return np.abs(count_out - count_gt)


def _f_count(s_out, s_gt):
    """Convert to count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        count_out: [B]
        count_gt: [B]
        num_obj: [B]
    """
    count_out = s_out.sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    num_obj = np.maximum(count_gt, 1)
    return count_out, count_gt, num_obj


def upsample(y_out, y_gt):
    """Upsample y_out into size of y_gt.

    Args:
        y_out: list of [T, H', W']
        y_gt: list of [T, H, W]

    Returns:
        y_out_resize: list of [T, H, W]
    """
    y_out_resize = []
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    for ii in xrange(num_ex):
        y_out_resize.append(np.zeros(y_gt[ii].shape, dtype='float32'))
        for jj in xrange(timespan):
            y_out_resize[ii][jj] = cv2.resize(
                y_out[ii][jj], (y_gt[ii].shape[2], y_gt[ii].shape[1]),
                interpolation=cv2.INTER_NEAREST)
    return y_out_resize


class StageAnalyzer(object):
    """Record average statistics."""

    def __init__(self, name, func, fname=None):
        self.avg = 0.0
        self.num_ex = 0
        self.name = name
        self.func = func

    def stage(self, y_out, y_gt, s_out, s_gt):
        """Record one batch."""
        _tmp = self.func(y_out, y_gt, s_out, s_gt).sum()
        _num = len(y_out)
        self.num_ex += _num
        self.avg += _tmp
        pass

    def finalize(self):
        """Finalize statistics."""
        self.avg /= self.num_ex
        log.info('{:20s}{:.4f}'.format(self.name, self.avg))
        pass


def run_eval(sess, m, dataset, batch_size=10, fname=None, cvppp_test=False):
    """Run evaluation

    Args:
        sess: tensorflow session
        m: model
        dataset: dataset object
        batch_size: mini-batch to run
        fname: output report filename
        cvppp_test: whether in test mode of CVPPP dataset
    """
    analyzers = []
    if not cvppp_test:
        analyzers = [StageAnalyzer('IOU', f_ins_iou, fname=fname),
                     StageAnalyzer('SBD', f_symmetric_best_dice, fname=fname),
                     StageAnalyzer('WT COV', f_wt_coverage, fname=fname),
                     StageAnalyzer('UNWT COV', f_unwt_coverage, fname=fname),
                     StageAnalyzer('FG DICE', f_fg_dice, fname=fname),
                     StageAnalyzer('FG IOU', f_fg_iou, fname=fname),
                     StageAnalyzer('COUNT ACC', f_count_acc, fname=fname),
                     StageAnalyzer('DIC', f_dic, fname=fname),
                     StageAnalyzer('|DIC|', f_dic_abs, fname=fname)]
    else:
        analyzers = [StageAnalyzer('FG DICE', f_fg_dice, fname=fname),
                     StageAnalyzer('FG IOU', f_fg_iou, fname=fname)]

    data = dataset.get_dataset()
    num_ex = data['input'].shape[0]
    batch_size = 10
    batch_iter = BatchIterator(num_ex,
                               batch_size=batch_size,
                               get_fn=get_batch_fn(data),
                               cycle=False,
                               progress_bar=True)
    _run_eval(sess, m, dataset, batch_iter, analyzers)
    pass


def _run_eval(sess, m, dataset, batch_iter, analyzers):
    output_list = [m['y_out'], m['s_out']]
    for x, y_gt, s_gt, idx in batch_iter:
        feed_dict = {m['x']: x, m['y_gt']: y_gt, m['phase_train']: False}
        r = sess.run(output_list, feed_dict)
        y_out, s_out = postprocess(r[0], r[1])
        y_gt = [_y_gt.astype('float32') for _y_gt in dataset.get_labels(idx)]
        y_out = upsample(y_out, y_gt)
        [analyzer.stage(y_out, y_gt, s_out, s_gt) for analyzer in analyzers]
        pass
    [analyzer.finalize() for analyzer in analyzers]
    pass
