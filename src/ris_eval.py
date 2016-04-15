from __future__ import division

import cslab_environ

import argparse
import os
import tensorflow as tf

from utils import logger
from utils.saver import Saver

import ris_attn_model as attn_model
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
    parser = argparse.ArgumentParser(description='Run evaluation')
    parser.add_argument('--dataset', default='cvppp')
    parser.add_argument('--model_id', default=None)
    parser.add_argument(
        '--results', default='/ais/gobi3/u/mren/results/img-count')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    tf.set_random_seed(1234)
    model_folder = os.path.join(args.results, args.model_id)
    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    step = ckpt_info['step']
    model_id = ckpt_info['model_id']

    dataset = base.get_dataset(args.dataset, data_opt)

    log.info('Building model')
    model = attn_model.get_model(model_opt)
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    for key in dataset:
        if args.dataset == 'cvppp' and key == 'test':
            output_only = True
        else:
            output_only = False
        log.info('Running {} set'.format(key))
        base.run_eval(sess, model, dataset[key], output_only=output_only)

    # # Test
    # sess = None
    # model = None
    # dataset = None
    # base.run_eval(sess, model, dataset)

    # Still need:
    # run eval in batch (not in total)
    # up sample images
    # output image labels
    pass
