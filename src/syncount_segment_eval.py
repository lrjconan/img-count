import sys
sys.path.insert(0, '/pkgs/tensorflow-cpu-0.5.0')

from utils import logger

import argparse
import cv2
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import syncount_gen_data as data
import syncount_segment as model
import tensorflow as tf

log = logger.get()


def _get_latest_ckpt(folder):
    """Get the latest checkpoint filename in a folder."""
    ckpt_fname_pattern = os.path.join(folder, 'model.ckpt-*')
    ckpt_fname_list = []
    for fname in os.listdir(folder):
        fullname = os.path.join(folder, fname)
        if fnmatch.fnmatch(fullname, ckpt_fname_pattern):
            ckpt_fname_list.append(fullname)
    if len(ckpt_fname_list) == 0:
        raise Exception('No checkpoint file found.')
    ckpt_fname_step = [int(fn.split('-')[-1]) for fn in ckpt_fname_list]
    latest_step = max(ckpt_fname_step)

    return os.path.join(folder, 'model.ckpt-{}'.format(latest_step))


def _plot_results(img, segm_label, segm_out, obj_label, obj_out, title=''):
    # Plot results
    num_img = img.shape[0]
    f, axarr = plt.subplots(num_img, 3)
    for ii in xrange(num_img):
        for jj in xrange(3):
            axarr[ii, jj].set_axis_off()
        axarr[ii, 0].imshow(img[ii])
        axarr[ii, 1].imshow(segm_label[ii])
        axarr[ii, 1].text(0, 0, '{:.2f}'.format(obj_label[ii, 0]), 
                          color=(0, 0, 0), size=8)
        axarr[ii, 2].imshow(segm_out[ii])
        axarr[ii, 2].text(0, 0, '{:.2f}'.format(obj_out[ii, 0]), 
                          color=(0, 0, 0), size=8)
    f.suptitle(title)


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

    ckpt_fname = _get_latest_ckpt(args.model)
    opt_fname = os.path.join(args.model, 'opt.pkl')

    with open(opt_fname, 'rb') as f_opt:
        opt = pkl.load(f_opt)
    log.info(opt)

    dataset = model.get_dataset(opt, 10, 10)

    img = dataset['valid']['input']
    segm_label = dataset['valid']['label_segmentation']
    obj_label = dataset['valid']['label_objectness']

    # Plot positive and negative examples separately.
    # Sample the first 20 postive and negative examples.
    num_ex = 10
    pos_idx = obj_label[:, 0] == 1
    neg_idx = obj_label[:, 0] == 0

    # Create model
    m = model.get_train_model(opt)

    # Create saver
    saver = tf.train.Saver(tf.all_variables())
    feed_dict_pos = {m['inp']: (img[pos_idx])[:num_ex]}
    feed_dict_neg = {m['inp']: (img[neg_idx])[:num_ex]}

    # Run model
    with tf.Session() as sess:
        # Restores from checkpoint
        saver.restore(sess, ckpt_fname)
        segm_out_pos = sess.run(m['segm'], feed_dict=feed_dict_pos)
        obj_out_pos = sess.run(m['obj'], feed_dict=feed_dict_pos)
        segm_out_neg = sess.run(m['segm'], feed_dict=feed_dict_neg)
        obj_out_neg = sess.run(m['obj'], feed_dict=feed_dict_neg)
        w_conv1 = sess.run(m['w_conv1'])

    # Plot positive examples
    _plot_results((img[pos_idx])[:num_ex], (segm_label[pos_idx])[:num_ex], 
                   segm_out_pos, (obj_label[pos_idx])[:num_ex], obj_out_pos, 
                   'Positive Examples')

    # Plot negative examples
    _plot_results((img[neg_idx])[:num_ex], (segm_label[neg_idx])[:num_ex], 
                   segm_out_neg, (obj_label[neg_idx])[:num_ex], obj_out_neg, 
                   'Negative Examples')

    # Normalize 1st layer filters and plot them
    w_conv1 = w_conv1 / np.max(w_conv1)
    num_filters = w_conv1.shape[3]
    num_rows = 4
    num_cols = num_filters / num_rows
    log.info('Number of filters: {}'.format(num_filters))
    f, axarr = plt.subplots(num_rows, num_cols)
    for ii in xrange(num_rows):
        for jj in xrange(num_cols):
            axarr[ii, jj].imshow(w_conv1[:, :, :, ii * num_cols + jj])
            axarr[ii, jj].set_axis_off()

    plt.show()
