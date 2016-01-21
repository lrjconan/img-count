from data_api import mnist
from utils import logger
import argparse
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import vae_mnist as model

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

    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)

    # Create model and load trained model.
    train_model = model.get_train_model(opt)
    sess = tf.Session()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, ckpt_fname)
    m_enc = model.get_encoder(opt, sess, train_model)
    m_dec = model.get_decoder(opt, sess, train_model)
    m_ae = model.get_autoencoder(opt, sess, train_model)

    # Generate random MNIST digits.
    num_row = 4
    num_col = 5
    random = np.random.RandomState(2)
    z = random.normal(0, 1, [num_row * num_col, opt['num_hid']])
    mu_x = sess.run(m_dec['mu_x'], feed_dict={m_dec['z']: z})

    f1, axarr = plt.subplots(num_row, num_col)
    for ii in xrange(num_row):
        for jj in xrange(num_col):
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(mu_x[idx].reshape([28, 28]), cmap=cm.Greys_r)
            axarr[ii, jj].set_axis_off()

    f2, axarr = plt.subplots(num_row, num_col)
    x = dataset.test.images[:num_row * num_col]
    x = model.preprocess(x, opt)
    
    x_out = sess.run(m_ae['mu_dec'], feed_dict={m_ae['x']: x})
    
    for ii in xrange(num_row):
        for jj in xrange(num_col):
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(x_out[idx].reshape([28, 28]), cmap=cm.Greys_r)
            axarr[ii, jj].set_axis_off()
    f2.suptitle('Output', fontsize=16)

    f3, axarr = plt.subplots(num_row, num_col)
    for ii in xrange(num_row):
        for jj in xrange(num_col):
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(x[idx].reshape([28, 28]), cmap=cm.Greys_r)
            axarr[ii, jj].set_axis_off()
    f3.suptitle('Input', fontsize=16)


    plt.show()
