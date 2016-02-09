import cslab_environ

from data_api import mnist
from utils.batch_iter import BatchIterator
from utils import logger
from utils import saver
import argparse
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import rec_ins_segm as model
import tensorflow as tf

log = logger.get()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate variational autoencoder on mnist')
    parser.add_argument('-model', default=None, help='Model save folder')
    parser.add_argument('-plot', action='store_true',
                        help='Whether to plot generated data.')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    ckpt_info = saver.get_ckpt_info(args.model)
    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    ckpt_fname = ckpt_info['ckpt_fname']

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    num_ex_valid = 1000
    dataset = model.get_dataset(data_opt, 10, num_ex_valid)
    sess = tf.Session()
    m = model.get_model(model_opt, device=device, train=False)

    saver.restore_ckpt(sess, ckpt_fname)

    num_ex_plot = 10
    x = dataset['valid']['input']
    y_gt = dataset['valid']['label_segmentation']
    s_gt = dataset['valid']['label_score']
    x, y_gt, s_gt = model.preprocess(x, y_gt, s_gt)

    if args.plot:
        log.info('Qualitative analysis')
        y_out, s_out = sess.run([m['y_out'], m['s_out']], feed_dict={
            m['x']: x[: num_ex_plot]
        })

        log.info(y_out.shape)
        log.info(s_out.shape)

        num_row = y_out.shape[0]
        num_col = y_out.shape[1] + 1

        f1, axarr = plt.subplots(num_row, num_col)

        for ii in xrange(num_row):
            for jj in xrange(num_col):
                axarr[ii, jj].set_axis_off()
                if jj == 0:
                    axarr[ii, jj].imshow(x[ii])
                else:
                    axarr[ii, jj].imshow(y_out[ii, jj - 1])
                    axarr[ii, jj].text(0, 0, '{:.2f}'.format(s_out[ii, jj - 1]),
                                       color=(0, 0, 0), size=8)

        plt.show()

    log.info('Quantatative analysis')
    y_out = np.zeros(y_gt.shape)

    def get_batch(start, end):
        return x[start: end], y_gt[start: end], s_gt[start: end]

    batch_size_valid = 100
    count_acc = 0
    iou = 0
    for x_bat, y_bat, s_bat in BatchIterator(num_ex_valid,
                                             batch_size=batch_size_valid,
                                             get_fn=get_batch,
                                             progress_bar=True):
        y_out, s_out = sess.run([m['y_out'], m['s_out']], feed_dict={
            m['x']: x_bat
        })
        count_out = (s_out > 0.5).astype('float').sum(axis=1)
        # log.info(count_out)
        # log.info(count_out.shape)
        count_gt = s_bat.sum(axis=1)
        # log.info(count_gt)
        # log.info(count_gt.shape)
        count_acc += (count_out == count_gt).astype('float').sum()
        # You need optimal matching to compute the IOU score.
        # y_out_hard = (y_out > 0.5).astype('float')
        # iou += ((y_out_hard * y_bat).sum(axis=1) / \
        #     (y_out_hard + y_bat - y_out_hard * y_bat).sum(axis=1)).sum()

    count_acc /= num_ex_valid
    iou /= num_ex_valid

    log.info('Count accuracy: {:.4f}'.format(count_acc))
    log.info('Average IOU: {:.4f}'.format(iou))

    sess.close()
