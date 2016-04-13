import cslab_environ

import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from data_api import cvppp

from utils import logger
from utils.saver import Saver

import ris_attn_model as attn_model
import ris_train_base as trainer
import ris_eval_base as base

log = logger.get()

"""
Eval script!!
We need to report score on the original size of the image!!

Input: downsampled size
Output: downsampled size
Output2: for kitti, query the original image size and then upsample.
for CVPPP, upsample to 500 x 530.
"""


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation + Attention')

    parser.add_argument('--dataset', default='cvppp')
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--results', default='/ais/gobi3/u/mren/results/img-count')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    tf.set_random_seed(1234)
    saver = None

    model_folder = os.path.join(args.results, args.model)

    log.info('Folder: {}'.format(model_folder))

    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    step = ckpt_info['step']
    model_id = ckpt_info['model_id']

    log.info('Building model')
    model = attn_model.get_model(model_opt)
    dataset = base.get_dataset(args.dataset, data_opt)
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    log.info('Running training set')
    res = base.run_inference(sess, model, dataset['train'], False)
    base.run_eval(res['y_out'], res['y_gt'], res['s_out'], res['s_gt'])

    log.info('Running validation set')
    res = run_inference(sess, model, dataset['valid'], False)
    run_eval(res['y_out'], res['y_gt'], res['s_out'], res['s_gt'])

    pass
