import sys
sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')

from utils import logger

import argparse
import cv2
import fnmatch
import os
import pickle as pkl
import syncount_gen_data as data
import syncount_segment as train
import tensorflow as tf

log = logger.get()


def get_latest_ckpt(folder):
    """Get the latest checkpoint filename in a folder."""
    ckpt_fname_pattern = os.path.join(folder, 'model.ckpt-*')
    ckpt_fname_list = []
    for fname in os.listdir(args.model):
        fullname = os.path.join(args.model, fname)
        if fnmatch.fnmatch(fullname, ckpt_fname_pattern):
            ckpt_fname_list.append(fullname)
    if len(ckpt_fname_list) == 0:
        raise Exception('No checkpoint file found.')
    ckpt_fname_step = [int(fn.split('-')[-1]) for fn in ckpt_fname_list]
    latest_step = max(ckpt_fname_step)

    return os.path.join(folder, 'model.ckpt-{}'.format(latest_step))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models on synthetic counting images')
    parser.add_argument('-model', default=None, help='Model save folder')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()

    if not args.model:
        log.fatal('You must provide model folder using -model.')

    ckpt_fname = get_latest_ckpt(args.model)
    opt_fname = os.path.join(args.model, 'opt.pkl')

    with open(opt_fname, 'rb') as f_opt:
        opt = pkl.load(f_opt)
    log.info(opt)

    dataset = train.get_dataset(opt, 10, 10)

    idx = 0
    img = dataset['valid']['input'][idx:idx + 1]
    label_segm = dataset['valid']['label_segmentation'][idx:idx + 1]
    label_obj = dataset['valid']['label_objectness'][idx:idx + 1]
    log.info('Image: {}'.format(img.shape))
    log.info('Label segmentation: {}'.format(label_segm.shape))
    cv2.imshow('img', img[0])

    # Create model
    m = train.create_model(opt)

    # Create saver
    saver = tf.train.Saver(tf.all_variables())
    feed_dict = {m['inp']: img,
                 m['segm_gt']: label_segm,
                 m['obj_gt']: label_obj}

    with tf.Session() as sess:
        # Restores from checkpoint
        saver.restore(sess, ckpt_fname)
        segm_out = sess.run(m['segm'], feed_dict=feed_dict)
        obj_out = sess.run(m['obj'], feed_dict=feed_dict)

        log.info('Output segmentation: {}'.format(segm_out.shape))
        log.info('Output segmentation: {}'.format(segm_out))
        cv2.imshow('output', segm_out[0])
        log.info('Output objectness: {}'.format(obj_out.shape))
        log.info('Output objectness: {}'.format(obj_out))

    cv2.imshow('label', label_segm[0])
    cv2.waitKey()
