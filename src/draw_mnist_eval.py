import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')
sys.path.insert(0, '/u/mren/code/img-count/third_party/tensorflow/_python_build/')

from data_api import mnist
from utils import logger
import argparse
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import draw_mnist as model

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


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate variational autoencoder on mnist')
    parser.add_argument('-model', default=None, help='Model save folder')
    args = parser.parse_args()

    return args


def sigmoid(a):
    return 1 / (1 + np.exp(-a + 1e-7))

if __name__ == '__main__':
    args = parse_args()
    log.log_args()

    if not args.model:
        log.fatal('You must provide model folder using -model.')

    ckpt_fname = _get_latest_ckpt(args.model)
    opt_fname = os.path.join(args.model, 'opt.pkl')

    # Load model configs.
    with open(opt_fname, 'rb') as f_opt:
        opt = pkl.load(f_opt)
    log.info(opt)
    if 'squash' not in opt:
        opt['squash'] = False

    # Load dataset.
    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)

    # Create model and load trained model.
    m_ae = model.get_model(opt, train=False)
    sess = tf.Session()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, ckpt_fname)
    # m_ae = model.get_autoencoder(opt, sess, train_model)

    # Generate reconstruct MNIST digits.
    num_row = 16
    num_col = opt['timespan']
    f1, axarr = plt.subplots(num_row, num_col)
    x = dataset.test.images[:num_row / 4].reshape(
        [-1, opt['inp_height'], opt['inp_width']])

    # Bernoulli only
    if opt['output_dist'] == 'Bernoulli':
        x = (x > 0.5).astype('float32')

    # Run model.
    results = m_ae['x_rec']
    results.extend(m_ae['ctr_x_r'])
    results.extend(m_ae['ctr_y_r'])
    results.extend(m_ae['delta_r'])
    results.extend(m_ae['lg_gamma_r'])
    results.extend(m_ae['readout_x'])

    results.extend(m_ae['ctr_x_w'])
    results.extend(m_ae['ctr_y_w'])
    results.extend(m_ae['delta_w'])
    results.extend(m_ae['lg_gamma_w'])
    results.extend(m_ae['canvas_delta'])

    results.extend(m_ae['filter_x_r'])
    results.extend(m_ae['filter_y_r'])
    results.extend(m_ae['mu_x_r'])
    results.extend(m_ae['mu_y_r'])
    results.extend(m_ae['lg_var_r'])

    results.extend(m_ae['filter_x_w'])
    results.extend(m_ae['filter_y_w'])
    results.extend(m_ae['mu_x_w'])
    results.extend(m_ae['mu_y_w'])
    results.extend(m_ae['lg_var_w'])

    r = sess.run(results, feed_dict={m_ae['x']: x})

    tt = opt['timespan']
    x_rec        = r[ 0 * tt:  1 * tt]
    
    ctr_x_r      = r[ 1 * tt:  2 * tt]
    ctr_y_r      = r[ 2 * tt:  3 * tt]
    delta_r      = r[ 3 * tt:  4 * tt]
    lg_gamma_r   = r[ 4 * tt:  5 * tt]
    readout_x    = r[ 5 * tt:  6 * tt]
    
    ctr_x_w      = r[ 6 * tt:  7 * tt]
    ctr_y_w      = r[ 7 * tt:  8 * tt]
    delta_w      = r[ 8 * tt:  9 * tt]
    lg_gamma_w   = r[ 9 * tt: 10 * tt]
    canvas_delta = r[10 * tt: 11 * tt]

    filter_x_r   = r[11 * tt: 12 * tt]
    filter_y_r   = r[12 * tt: 13 * tt]
    mu_x_r       = r[13 * tt: 14 * tt]
    mu_y_r       = r[14 * tt: 15 * tt]
    lg_var_r     = r[15 * tt: 16 * tt]

    filter_x_w   = r[16 * tt: 17 * tt]
    filter_y_w   = r[17 * tt: 18 * tt]
    mu_x_w       = r[18 * tt: 19 * tt]
    mu_y_w       = r[19 * tt: 20 * tt]
    lg_var_w     = r[20 * tt: 21 * tt]

    #
    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            idx = ii / 4
            if ii % 4 == 0:
                # Plot canvas image.
                axarr[ii, jj].imshow(x_rec[jj][idx], cmap=cm.Greys_r)
                # Plot write attention controller box.
                top_left_x = (ctr_x_w[jj][idx] -
                              delta_w[jj][idx] *
                              (opt['filter_size_w'] - 1) / 2.0)
                top_left_y = (ctr_y_w[jj][idx] -
                              delta_w[jj][idx] *
                              (opt['filter_size_w'] - 1) / 2.0)
                axarr[ii, jj].add_patch(
                    patches.Rectangle(
                        (top_left_x, top_left_y),
                        delta_w[jj][idx] * (opt['filter_size_w'] - 1),
                        delta_w[jj][idx] * (opt['filter_size_w'] - 1),
                        fill=False,
                        color='r')
                )
            elif ii % 4 == 1:
                # Plot write out image.
                axarr[ii, jj].imshow(sigmoid(canvas_delta[jj][idx]),
                                     cmap=cm.Greys_r)
            elif ii % 4 == 2:
                # Plot read out image.
                log.info('Time: {}'.format(jj))
                axarr[ii, jj].imshow(
                    readout_x[jj][idx] / np.exp(lg_gamma_r[jj][idx]),
                    cmap=cm.Greys_r)
                log.info('Read head centre: {}'.format(
                    (ctr_x_r[jj][idx], ctr_y_r[jj][idx])))
                log.info('Read head stride: {}'.format(delta_r[jj][idx]))
                log.info('Mu x read: {}'.format(mu_x_r[jj][idx]))
                log.info('Mu y read: {}'.format(mu_y_r[jj][idx]))
                log.info('Var: {}'.format(np.exp(lg_var_r[jj][idx])))
                log.info('Filter x read: {}'.format(filter_x_r[jj][idx]))
                log.info('Filter y read: {}'.format(filter_y_r[jj][idx]))
                log.info('Read out: {}'.format(readout_x[jj][idx]))
                log.info('Gamma: {}'.format(np.exp(lg_gamma_r[jj][idx])))
            elif ii % 4 == 3:
                # Plot original image.
                axarr[ii, jj].imshow(x[idx], cmap=cm.Greys_r)
                # Plot read attention controller box.
                top_left_x = (ctr_x_r[jj][idx] -
                              delta_r[jj][idx] *
                              (opt['filter_size_r'] - 1) / 2.0)
                top_left_y = (ctr_y_r[jj][idx] -
                              delta_r[jj][idx] *
                              (opt['filter_size_r'] - 1) / 2.0)
                axarr[ii, jj].add_patch(
                    patches.Rectangle(
                        (top_left_x, top_left_y),
                        delta_r[jj][idx] * (opt['filter_size_r'] - 1),
                        delta_r[jj][idx] * (opt['filter_size_r'] - 1),
                        fill=False,
                        color='r')
                )
                axarr[ii, jj].set_axis_off()

    plt.show()
