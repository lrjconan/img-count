"""
This code implements VAE (Variational Autoencoder) [1] on occluded MNIST.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage: python vae_mnist_half.py

Reference:
[1] D.P. Kingma, M. Welling. Auto-Encoding Variational Bayes. ICLR 2014.
"""

from data_api import mnist
from utils import logger
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import numpy as np
import os
import pickle as pkl
import tensorflow as tf

log = logger.get()


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_autoencoder(opt, sess, train_model):
    num_inp = opt['num_inp']

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


def get_encoder(opt, sess, train_model):
    num_inp = opt['num_inp']
    nl = eval(opt['non_linear'])

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


def get_decoder(opt, sess, train_model):
    num_hid = opt['num_hid']
    nl = eval(opt['non_linear'])

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


def get_train_model(opt):
    num_inp = opt['num_inp']
    num_hid_enc = opt['num_hid_enc']
    num_hid = opt['num_hid']
    num_hid_dec = opt['num_hid_dec']
    wd = opt['weight_decay']
    nl = eval(opt['non_linear'])

    # Occluded input (N, D)
    x = tf.placeholder('float', [None, num_inp], name='x')
    # Full input (groundtruth) (N, D)
    x_ = tf.placeholder('float', [None, num_inp], name='x_')

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
    log_sigma_enc = tf.add(tf.matmul(h_enc, w_3), b_3, name='log_sigma_enc')

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
                                (x_ - mu_dec) / sigma_dec *
                                (x_ - mu_dec) / sigma_dec,
                                name='log_pxz')
    elif opt['output_dist'] == 'Bernoulli':
        # Bernoulli posterior: p(x | z), (same as cross entropy)
        log_pxz = tf.reduce_sum(x_ * tf.log(mu_dec + 1e-7) +
                                (1 - x_) * tf.log((1 - mu_dec + 1e-7)),
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
    train_step = tf.train.AdamOptimizer(lr, epsilon=eps).minimize(total_loss)

    m = {
        'x': x,
        'x_': x_,
        't': t,
        'w_1': w_1,
        'b_1': b_1,
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


def save_ckpt(folder, sess, opt, global_step=None):
    """Save checkpoint.

    Args:
        folder:
        sess:
        global_step:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    ckpt_path = os.path.join(
        folder, 'model.ckpt'.format(model_id))
    log.info('Saving checkpoint to {}'.format(ckpt_path))
    saver.save(sess, ckpt_path, global_step=global_step)
    opt_path = os.path.join(folder, 'opt.pkl')
    with open(opt_path, 'wb') as f_opt:
        pkl.dump(opt, f_opt)

    pass


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
        x_ = (x > 0.5).astype('float32').reshape([-1, 28 * 28])
        x2 = (x > 0.5).astype('float32').reshape([-1, 28, 28])
        # Lower half is occluded.
        x2[:, 14:, :] = 0.0
        x2 = x2.reshape([-1, 28 * 28])
    else:
        x_ = x.astype('float32').reshape([-1, 28 * 28])
        x2 = x.astype('float32').reshape([-1, 28, 28])
        # Lower half is occluded.
        x2[:, 14:, :] = 0.0
        x2 = x2.reshape([-1, 28 * 28])

    return x2, x_

if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    log.log_args()

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
    m = get_train_model(opt)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    task_name = 'vae_mnist_half'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    results_folder = args.results
    logs_folder = args.logs
    exp_folder = os.path.join(results_folder, model_id)
    exp_logs_folder = os.path.join(logs_folder, model_id)

    # Create time series logger
    train_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'train_logp.csv'), 'train_logp',
        buffer_size=25)
    valid_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'valid_logp.csv'), 'valid_logp',
        buffer_size=2)
    log.info(
        'Curves can be viewed at: http://{}/visualizer?id={}'.format(
            args.localhost, model_id))

    random = np.random.RandomState(2)
    step = 0
    while step < loop_config['num_steps']:

        # Validation
        valid_log_px_lb = 0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x, x_ = preprocess(batch[0], opt)
            t = random.normal(0, 1, [x.shape[0], opt['num_hid']])
            log_px_lb = sess.run(m['log_px_lb'], feed_dict={
                m['x']: x,
                m['x_']: x_,
                m['t']: t
            })
            valid_log_px_lb += log_px_lb * 100 / 10000.0
        log.info('step {:d}, valid logp {:.4f}'.format(step, valid_log_px_lb))
        valid_logger.add(step, valid_log_px_lb)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x, x_ = preprocess(batch[0], opt)
            t = random.normal(0, 1, [x.shape[0], opt['num_hid']])
            if step % 10 == 0:
                log_px_lb = sess.run(m['log_px_lb'], feed_dict={
                    m['x']: x,
                    m['x_']: x_,
                    m['t']: t
                })
                log.info('step {:d}, train logp {:.4f}'.format(
                    step, log_px_lb))
                train_logger.add(step, log_px_lb)

            sess.run(m['train_step'], feed_dict={
                m['x']: x,
                m['x_']: x_,
                m['t']: random.normal(0, 1, [x.shape[0], opt['num_hid']])
            })

            step += 1

            # Save model
            if step % args.steps_per_ckpt == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)
