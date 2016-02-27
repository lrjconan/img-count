"""
This code implements VAE (Variational Autoencoder) [1] on MNIST.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage: python vae_mnist.py

Reference:
[1] D.P. Kingma, M. Welling. Auto-encoding variational Bayes. ICLR 2014.
"""
import cslab_environ

from data_api import mnist
from utils import log_manager
from utils import logger
from utils import saver
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_autoencoder(opt, sess, train_model, device='/cpu:0'):
    num_inp = opt['num_inp']

    with tf.device(device):
        # Input (N, D)
        x = tf.placeholder('float', [None, num_inp])

        # Encoder hidden layer (N, H1)
        w_1 = tf.constant(sess.run(train_model['w_1']))
        b_1 = tf.constant(sess.run(train_model['b_1']))
        # h_enc = tf.tanh(tf.matmul(x, w_1) + b_1)
        h_enc = tf.nn.softplus(tf.matmul(x, w_1) + b_1, name='h_enc')

        # Encoder output: distribution parameters mu, log_sigma (N, 1, H)
        w_2 = tf.constant(sess.run(train_model['w_2']))
        b_2 = tf.constant(sess.run(train_model['b_2']))
        mu_enc = tf.add(tf.matmul(h_enc, w_2), b_2, name='mu_enc')

        # Encoder latent variable (N * M, H)
        z = mu_enc

        # Decoder hidden layer
        w_4 = tf.constant(sess.run(train_model['w_4']))
        b_4 = tf.constant(sess.run(train_model['b_4']))
        # h_dec = tf.tanh(tf.matmul(z, w_4) + b_4, name='h_dec')
        h_dec = tf.nn.softplus(tf.matmul(z, w_4) + b_4, name='h_dec')

        # Decoder output: distribution parameters mu, log_sigma
        w_5 = tf.constant(sess.run(train_model['w_5']))
        b_5 = tf.constant(sess.run(train_model['b_5']))
        mu_dec = tf.sigmoid(tf.matmul(h_dec, w_5) + b_5, name='mu_dec')

    return {
        'x': x,
        'w_1': w_1,
        'b_1': b_1,
        'w_2': w_2,
        'b_2': b_2,
        'w_4': w_4,
        'b_4': b_4,
        'w_5': w_5,
        'b_5': b_5,
        'z': z,
        'mu_dec': mu_dec
    }


def get_encoder(opt, sess, train_model, device='/cpu:0'):
    num_inp = opt['num_inp']
    nl = eval(opt['non_linear'])

    with tf.device(device):
        x = tf.placeholder('float', [None, num_inp], name='x')

        # Encoder hidden layer (N, H1)
        w_1 = tf.constant(sess.run(train_model['w_1']), name='w_1')
        b_1 = tf.constant(sess.run(train_model['b_1']), name='b_1')
        h_enc = nl(tf.matmul(x, w_1) + b_1, name='h_enc')

        # Encoder output: distribution parameters mu, log_sigma (N, 1, H)
        w_2 = tf.constant(sess.run(train_model['w_2']), name='w_2')
        b_2 = tf.constant(sess.run(train_model['b_2']), name='b_2')
        mu_z = tf.add(tf.matmul(h_enc, w_2), b_2, name='mu_z')

        m = {
            'x': x,
            'mu_z': mu_z
        }

        if opt['output_dist'] == 'Gaussian':
            w_3 = tf.constant(sess.run(train_model['w_3']))
            b_3 = tf.constant(sess.run(train_model['w_3']))
            sigma_z = tf.exp(0.5 * tf.matmul(h_enc, w_3) + b_3)
            m['sigma_z'] = sigma_z

    return m


def get_decoder(opt, sess, train_model, device='/cpu:0'):
    num_hid = opt['num_hid']
    nl = eval(opt['non_linear'])

    with tf.device(device):
        z = tf.placeholder('float', [None, num_hid], name='z')

        # Decoder hidden layer
        w_4 = tf.constant(sess.run(train_model['w_4']), name='w_4')
        b_4 = tf.constant(sess.run(train_model['b_4']), name='b_4')
        h_dec = nl(tf.matmul(z, w_4) + b_4, name='h_dec')

        # Decoder output: distribution parameters mu, log_sigma
        w_5 = tf.constant(sess.run(train_model['w_5']), name='w_5')
        b_5 = tf.constant(sess.run(train_model['b_5']), name='b_5')
        mu_x = tf.sigmoid(tf.matmul(h_dec, w_5) + b_5, name='mu_x')

        m = {
            'z': z,
            'mu_x': mu_x
        }

        if opt['output_dist'] == 'Gaussian':
            w_6 = tf.constant(sess.run(train_model['w_6']))
            b_6 = tf.constant(sess.run(train_model['b_6']))
            sigma_x = tf.exp(0.5 * (tf.matmul(h_dec, w_6) + b_6))
            m['sigma_x'] = sigma_x

    return m


def get_decoder_2(opt, train_model, device='/cpu:0'):
    num_hid = opt['num_hid']
    nl = eval(opt['non_linear'])

    with tf.device(device):
        z = tf.placeholder('float', [None, num_hid], name='z')

        # Decoder hidden layer
        w_4 = train_model['w_4']
        b_4 = train_model['b_4']
        h_dec = nl(tf.matmul(z, w_4) + b_4, name='h_dec')

        # Decoder output: distribution parameters mu, log_sigma
        w_5 = train_model['w_5']
        b_5 = train_model['b_5']
        mu_x = tf.sigmoid(tf.matmul(h_dec, w_5) + b_5, name='mu_x')

        m = {
            'z': z,
            'mu_x': mu_x
        }

        if opt['output_dist'] == 'Gaussian':
            w_6 = train_model['w_6']
            b_6 = train_model['b_6']
            sigma_x = tf.exp(0.5 * (tf.matmul(h_dec, w_6) + b_6))
            m['sigma_x'] = sigma_x

    return m


def get_train_model(opt, device='/cpu:0'):
    num_inp = opt['num_inp']
    num_hid_enc = opt['num_hid_enc']
    num_hid = opt['num_hid']
    num_hid_dec = opt['num_hid_dec']
    wd = opt['weight_decay']
    nl = eval(opt['non_linear'])

    with tf.device(device):
        # Input (N, D)
        x = tf.placeholder('float', [None, num_inp], name='x')

        # Encoder hidden layer (N, H1)
        w_1 = weight_variable([num_inp, num_hid_enc], wd=wd, name='w_1')
        b_1 = weight_variable([num_hid_enc], wd=wd, name='b_1')
        h_enc = nl(tf.matmul(x, w_1) + b_1, name='h_enc')

        # Encoder output: distribution parameters mu, log_sigma (N, 1, H)
        w_2 = weight_variable([num_hid_enc, num_hid], wd=wd, name='w_2')
        b_2 = weight_variable([num_hid], wd=wd, name='b_2')
        mu_enc = tf.matmul(h_enc, w_2) + b_2

        w_3 = weight_variable([num_hid_enc, num_hid], wd=wd, name='w_3')
        b_3 = weight_variable([num_hid], wd=wd, name='b_3')
        log_sigma_enc = tf.add(tf.matmul(h_enc, w_3),
                               b_3, name='log_sigma_enc')

        # Noise (N, M, H)
        t = tf.placeholder('float', [None, num_hid], name='t')

        # Encoder latent variable (N * M, H)
        z = tf.add(mu_enc, tf.mul(tf.exp(log_sigma_enc), t), name='z')

        # KL Divergence
        kl_qzx_pz = tf.mul(-0.5,
                           tf.reduce_sum(1 + 2 * log_sigma_enc - mu_enc * mu_enc -
                                         tf.exp(2 * log_sigma_enc)),
                           name='kl_qzx_pz')

        # Decoder hidden layer
        w_4 = weight_variable([num_hid, num_hid_dec], wd=wd, name='w_4')
        b_4 = weight_variable([num_hid_dec], wd=wd, name='b_4')
        h_dec = nl(tf.matmul(z, w_4) + b_4, name='h_dec')

        # Decoder output: distribution parameters mu, log_sigma
        w_5 = weight_variable([num_hid_dec, num_inp], wd=wd, name='w_5')
        b_5 = weight_variable([num_inp], wd=wd, name='b_5')
        mu_dec = tf.sigmoid(tf.matmul(h_dec, w_5) + b_5)

        # Gaussian posterior: p(x | z)
        if opt['output_dist'] == 'Gaussian':
            w_6 = weight_variable([num_hid_dec, num_inp], wd=wd, name='w_6')
            b_6 = weight_variable([num_inp], wd=wd, name='b_6')
            log_sigma_dec = tf.add(tf.matmul(h_dec, w_6),
                                   b_6, name='log_sigma_dec')
            sigma_dec = tf.exp(log_sigma_dec + 1e-4, name='sigma_dec')
            log_pxz = tf.reduce_sum(-0.5 * tf.log(2 * np.pi) - log_sigma_dec - 0.5 *
                                    (x - mu_dec) / sigma_dec *
                                    (x - mu_dec) / sigma_dec,
                                    name='log_pxz')
        elif opt['output_dist'] == 'Bernoulli':
            # Bernoulli posterior: p(x | z), (same as cross entropy)
            log_pxz = tf.reduce_sum(x * tf.log(mu_dec + 1e-7) +
                                    (1 - x) * tf.log((1 - mu_dec + 1e-7)),
                                    name='log_pxz')
        else:
            raise Exception(
                'Unknown output distribution type: {}'.format(opt['output_dist']))

        # Normalize by number of examples
        num_ex = tf.shape(x, name='num_ex')

        # Variational lower bound of marginal log-likelihood
        w_kl = 1.0
        w_logp = 1.0
        log_px_lb = (-w_kl * kl_qzx_pz + w_logp * log_pxz) / \
            (w_kl + w_logp) * 2.0 / tf.to_float(num_ex[0])
        tf.add_to_collection('losses', -log_px_lb)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        lr = 1e-4
        eps = 1e-7
        train_step = tf.train.AdamOptimizer(lr, epsilon=eps).minimize(
            total_loss)

    m = {
        'x': x,
        't': t,
        'w_1': w_1,
        'b_1': b_1,
        'h_enc': h_enc,
        'w_2': w_2,
        'b_2': b_2,
        'w_3': w_3,
        'b_3': b_3,
        'mu_enc': mu_enc,
        'log_sigma_enc': log_sigma_enc,
        'z': z,
        'kl_qzx_pz': kl_qzx_pz,
        'w_4': w_4,
        'b_4': b_4,
        'h_dec': h_dec,
        'w_5': w_5,
        'b_5': b_5,
        'mu_dec': mu_dec,
        'log_pxz': log_pxz,
        'log_px_lb': log_px_lb,
        'train_step': train_step
    }

    if opt['output_dist'] == 'Gaussian':
        m['w_6'] = w_6
        m['b_6'] = b_6
        m['log_sigma_dec'] = log_sigma_dec

    return m


def parse_args():
    """Parse input arguments."""
    # Number of steps
    kNumSteps = 500000
    # Number of steps per checkpoint
    kStepsPerCkpt = 1000
    parser = argparse.ArgumentParser(
        description='Train DRAW')
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-results', default='../results',
                        help='Model results folder')
    parser.add_argument('-logs', default='../results',
                        help='Training curve logs folder')
    parser.add_argument('-localhost', default='localhost',
                        help='Local domain name')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    args = parser.parse_args()

    return args


def preprocess(x, opt):
    if opt['output_dist'] == 'Bernoulli':
        x2 = (x > 0.5).astype('float32').reshape([-1, 28 * 28])
    else:
        x2 = x.reshape([-1, 28 * 28])

    return x2


def get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


def plot_digits(fname, data, num_row, num_col):
    f, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))

    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            idx = ii * num_col + jj
            axarr[ii, jj].imshow(data[idx], cmap=cm.Greys_r)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=80)

    pass


def plot_decoder(fname, x, x_rec, num_row, num_col):
    f, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))

    for ii in xrange(num_row):
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            idx = ii * num_col + jj
            idx2 = idx / 2
            if idx % 2 == 0:
                axarr[ii, jj].imshow(
                    x[idx2].reshape([28, 28]), cmap=cm.Greys_r)
            else:
                axarr[ii, jj].imshow(
                    x_rec[idx2].reshape([28, 28]), cmap=cm.Greys_r)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=80)

    pass


if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()

    # Model ID
    model_id = get_model_id('vae_mnist')
    results_folder = args.results
    exp_folder = os.path.join(results_folder, model_id)

    # Logger
    if args.logs:
        logs_folder = args.logs
        logs_folder = os.path.join(logs_folder, model_id)

        log = logger.get(os.path.join(logs_folder, 'raw'))

        # Create time series logger
        logp_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'logp.csv'), ['train logp', 'valid logp'],
            name='Log prob',
            buffer_size=1)
        henc_sparsity_logger = TimeSeriesLogger(
            os.path.join(logs_folder,
                         'henc_sparsity.csv'), 'henc sparsity',
            name='Encoder hidden activation sparsity',
            buffer_size=1)
        hdec_sparsity_logger = TimeSeriesLogger(
            os.path.join(logs_folder,
                         'hdec_sparsity.csv'), 'hdec sparsity',
            name='Decoder hidden activation sparsity',
            buffer_size=1)
        step_time_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
            buffer_size=10)
        w1_image_fname = os.path.join(logs_folder, 'w1.png')
        decoder_image_fname = os.path.join(logs_folder, 'decoder.png')
        gen_image_fname = os.path.join(logs_folder, 'gen.png')
        registered_image = False
        log_manager.register(log.filename, 'plain', 'Raw logs')

        log.info(
            'Curves can be viewed at: http://{}/visualizer?id={}'.format(
                args.localhost, model_id))
    else:
        log = logger.get()

    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    opt = {
        'num_inp': 28 * 28,
        'num_hid_enc': 100,
        'num_hid': 20,
        'num_hid_dec': 100,
        'output_dist': 'Bernoulli',  # Bernoulli or Gaussian
        # 'output_dist': 'Gaussian',
        'non_linear': 'tf.nn.relu',
        # 'non_linear': 'tf.nn.tanh',
        'weight_decay': 5e-5
    }

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)
    m = get_train_model(opt, device=device)
    m_dec = get_decoder_2(opt, m, device=device)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    random = np.random.RandomState(2)
    step = 0
    while step < loop_config['num_steps']:

        # Validation
        valid_log_px_lb = 0.0
        henc_sparsity = 0.0
        hdec_sparsity = 0.0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x = preprocess(batch[0], opt)
            t = random.normal(0, 1, [x.shape[0], opt['num_hid']])
            log_px_lb, henc, hdec = sess.run([
                m['log_px_lb'],
                m['h_enc'],
                m['h_dec']],
                feed_dict={
                m['x']: x,
                m['t']: t
            })
            henc_sparsity += ((henc == 0.0).astype('float').sum() /
                              float(henc.size) / 100.0)
            hdec_sparsity += ((hdec == 0.0).astype('float').sum() /
                              float(hdec.size) / 100.0)
            valid_log_px_lb += log_px_lb / 100.0
        log.info('step {:d}, valid logp {:.4f}'.format(step, valid_log_px_lb))

        if args.logs:
            num_plot = 50
            x = dataset.test.images[: num_plot]
            t = random.normal(0, 1, [x.shape[0], opt['num_hid']])
            w1, x_rec = sess.run([m['w_1'], m['mu_dec']], feed_dict={
                m['x']: x,
                m['t']: t
            })

            z = random.normal(0, 1, [x.shape[0], opt['num_hid']])
            x_ = sess.run(m_dec['mu_x'], feed_dict={m_dec['z']: z})

            plot_digits(w1_image_fname, w1.transpose().reshape([-1, 28, 28]),
                        num_row=3, num_col=10)
            plot_decoder(decoder_image_fname, x, x_rec, num_row=3, num_col=10)
            plot_digits(gen_image_fname, x_.reshape([-1, 28, 28]),
                        num_row=3, num_col=10)

            if not registered_image:
                log_manager.register(
                    w1_image_fname, 'image', 'W1 visualization')
                log_manager.register(
                    decoder_image_fname, 'image', 'Decoder reconstruction')
                log_manager.register(
                    gen_image_fname, 'image', 'Generated digits')
                registered_image = True
            logp_logger.add(step, ['', valid_log_px_lb])
            henc_sparsity_logger.add(step, henc_sparsity)
            hdec_sparsity_logger.add(step, hdec_sparsity)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = preprocess(batch[0], opt)
            t = random.normal(0, 1, [x.shape[0], opt['num_hid']])

            st = time.time()
            r = sess.run([m['log_px_lb'], m['train_step']], feed_dict={
                m['x']: x,
                m['t']: random.normal(0, 1, [x.shape[0], opt['num_hid']])
            })
            if step % 10 == 0:
                log_px_lb = r[0]
                step_time = (time.time() - st) * 1000
                log.info('{:d} logp {:.4f} t {:.2f}ms'.format(
                    step, log_px_lb, step_time))
                logp_logger.add(step, [log_px_lb, ''])
                step_time_logger.add(step, step_time)

            step += 1

            # # Save model
            # if step % args.steps_per_ckpt == 0:
            #     saver.save_ckpt(exp_folder, sess, model_opt=opt,
            #                     global_step=step)

    sess.close()
