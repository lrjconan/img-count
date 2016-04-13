import cslab_environ

import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from data_api import cvppp

from utils.batch_iter import BatchIterator


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


def get_batch_fn(dataset):
    """
    Preprocess mini-batch data given start and end indices.
    """
    def get_batch(idx):
        x_bat = dataset['input'][idx]
        y_bat = dataset['label_segmentation'][idx]
        s_bat = dataset['label_score'][idx]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)

        return x_bat, y_bat, s_bat

    return get_batch


def run_inference(sess, m, dataset, phase_train):
    output_list = [m['y_out'], m['s_out']]

    num_ex = dataset['input'].shape[0]
    batch_size = 10

    batch_iter = BatchIterator(num_ex,
                               batch_size=batch_size,
                               get_fn=get_batch_fn(dataset),
                               cycle=False,
                               progress_bar=True)
    y_out = None
    s_out = None
    y_gt = None
    s_gt = None
    count = 0

    for x, y, s in batch_iter:
        r = sess.run(output_list, feed_dict={
                     m['x']: x,
                     m['y_gt']: y,
                     # m['s_gt']: s,
                     m['phase_train']: phase_train}
                     )
        _y_out = r[0]
        _s_out = r[1]
        bat_sz = _y_out.shape[0]
        if y_out is None:
            y_out = np.zeros(
                [num_ex, _y_out.shape[1], _y_out.shape[2], _y_out.shape[3]])
            y_gt = np.zeros(y_out.shape)
            s_out = np.zeros([num_ex, _y_out.shape[1]])
            s_gt = np.zeros(s_out.shape)
        y_gt[count: count + bat_sz] = y
        s_gt[count: count + bat_sz] = s
        y_out[count: count + bat_sz] = _y_out
        s_out[count: count + bat_sz] = _s_out

        count += bat_sz

    return {
        'y_gt': y_gt,
        's_gt': s_gt,
        'y_out': y_out,
        's_out': s_out
    }


def best_dice(a, b, num_obj):
    bd = np.zeros([a.shape[0], a.shape[1]])

    for ii in xrange(a.shape[1]):
        a_ = a[:, ii: ii + 1, :, :]
        card_a = a_.sum(axis=3).sum(axis=2)
        card_b = b.sum(axis=3).sum(axis=2)
        card_ab = (a_ * b).sum(axis=3).sum(axis=2)
        dice = 2 * card_ab / (card_a + card_b + 1e-5)
        bd[:, ii] = dice.max(axis=1)

    return bd.sum(axis=1) / num_obj


def symmetric_best_dice(y_out, y_gt, num_obj):
    # num_ex = y_out.shape[0]
    num_obj = y_gt.shape[1]
    bd1 = best_dice(y_out, y_gt, num_obj)
    bd2 = best_dice(y_gt, y_out, num_obj)
    print 'bd1', bd1
    print 'bd2', bd2

    return np.minimum(bd1, bd2)


def coverage(y_out, y_gt, num_obj, weighted=False):
    cov = np.zeros([y_out.shape[0], y_out.shape[1]])
    for ii in xrange(y_gt.shape[1]):
        y_gt_ = y_gt[:, ii: ii + 1, :, :]
        iou_ii = iou(y_out, y_gt_)
        cov[:, ii] = iou_ii.max(axis=1)

    if weighted:
        print y_gt.shape
        print y_gt.sum(axis=3).sum(axis=2).sum(axis=1)
        weights = y_gt.sum(axis=3).sum(axis=2) / \
            y_gt.sum(axis=3).sum(axis=2).sum(axis=1, keepdims=True)
        cov *= weights

    return cov.sum(axis=1) / num_obj


def iou(a, b):
    inter = (a * b).sum(axis=3).sum(axis=2)
    union = (a + b).sum(axis=3).sum(axis=2) - inter

    return inter / (union + 1e-5)


def build_matching():
    pass


def run_eval(y_out, y_gt, s_out, s_gt):
    s_mask = np.reshape(s_out, [-1, s_out.shape[1], 1, 1])
    y_out = y_out * s_mask
    y_out_max = np.argmax(y_out, axis=1)

    y_out_hard = np.zeros(y_out.shape)
    for idx in xrange(y_out.shape[1]):
        y_out_hard[:, idx] = (y_out_max == idx).astype(
            'float') * (y_out[:, idx] > 0.5).astype('float')

    y_out_hard = (y_out == y_out_max).astype('float')
    y_out_hard = (y_out_hard > 0.5).astype('float')
    count_out = (s_out > 0.5).astype('float').sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    num_obj = np.maximum(count_gt, 1)
    print 'num obj', num_obj

    # # Upsample the results
    # height = y_out.shape[2]
    # width = y_out.shape[3]
    # y_out = y_out.reshape([-1, height, width])
    # y_out_hi = np.zeros([y_out.shape[0], height, width])

    # for ii in xrange(y_out.shape[0]):
    #     y_out_hi[ii] = cv2.resize(
    #         y_out[ii], (500, 530), interpolation=cv2.INTER_NEAREST)

    sbd = symmetric_best_dice(y_out, y_gt, num_obj).mean()
    unwt_cov = coverage(y_out, y_gt, num_obj, weighted=False).mean()
    wt_cov = coverage(y_out, y_gt, num_obj, weighted=True).mean()

    count_acc = (count_out == count_gt).astype('float').mean()
    print 'correct', (count_out == count_gt)
    dic = (count_out - count_gt).mean()
    dic_abs = np.abs(count_out - count_gt).mean()

    log.info('{:10s}{:.4f}'.format('SBD', sbd))
    log.info('{:10s}{:.4f}'.format('Wt Cov', wt_cov))
    log.info('{:10s}{:.4f}'.format('Unwt Cov', unwt_cov))

    log.info('{:10s}{:.4f}'.format('Count Acc', count_acc))
    log.info('{:10s}{:.4f}'.format('DiC', dic))
    log.info('{:10s}{:.4f}'.format('|DiC|', dic_abs))

    pass
