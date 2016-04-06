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

import nnlib as nn


def get_model(opt, device='/cpu:0'):
    with tf.device(device):
        # Input (N, D)
        x = tf.placeholder('float', [None, 28 * 28])
        y_gt = tf.placeholder('float', [None, 10])
        phase_train = tf.placeholder('bool')
        x = tf.reshape(x, [-1, 28, 28, 1])

        cnn_f = [5] + [3, 3, 3, 3] * 4
        cnn_ch = [1] + [8] + [8, 8, 8, 8] + [16, 16, 16, 16] + \
            [32, 32, 32, 32] + [64, 64, 64, 64]
        cnn_pool = [1] + [1, 1, 1, 2] * 4
        cnn_res = [0] + [0, 2, 0, 2] * 4
        cnn_act = [tf.nn.relu] * 17
        cnn_use_bn = [True] * 17

        cnn = nn.res_cnn(cnn_f, cnn_ch, cnn_pool, cnn_res,
                         cnn_act, cnn_use_bn, phase_train=phase_train)
        h = cnn(x)
        h = tf.squeeze(nn.avg_pool(h[-1], 2))
        w = nn.weight_variable([64, 10])
        b = nn.weight_variable([10])
        y_out = tf.nn.softmax(tf.matmul(h, w) + b)
        num_ex_f = tf.to_float(tf.shape(x)[0])
        ce = -tf.reduce_sum(y_gt * tf.log(y_out + 1e-5)) / num_ex_f
        correct = tf.equal(tf.argmax(y_gt, 1), tf.argmax(y_out, 1))
        acc = tf.reduce_sum(tf.to_float(correct)) / num_ex_f

        lr = 1e-3
        eps = 1e-7
        train_step = tf.train.AdamOptimizer(lr, epsilon=eps).minimize(ce)

    m = {
        'x': x,
        'y_gt': y_gt,
        'y_out': y_out,
        'phase_train': phase_train,
        'train_step': train_step,
        'ce': ce,
        'acc': acc
    }

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


def preprocess(x):
    x2 = x.reshape([-1, 28, 28, 1])

    return x2


def get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()

    # Model ID
    model_id = get_model_id('residual_mnist')
    results_folder = args.results
    exp_folder = os.path.join(results_folder, model_id)

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)
    m = get_model(None, device=device)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Logger
    if args.logs:
        logs_folder = args.logs
        logs_folder = os.path.join(logs_folder, model_id)

        log = logger.get(os.path.join(logs_folder, 'raw'))

        # Create time series logger
        ce_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'ce.csv'), ['train', 'valid'],
            name='Cross Entropy',
            buffer_size=1)
        acc_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'acc.csv'), ['train', 'valid'],
            name='Accuracy',
            buffer_size=1)
        step_time_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
            buffer_size=10)

        log_manager.register(log.filename, 'plain', 'Raw logs')
        log.info(
            'Curves can be viewed at: http://{}/deep-dashboard?id={}'.format(
                args.localhost, model_id))
    else:
        log = logger.get()

    log.log_args()

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    random = np.random.RandomState(2)
    step = 0
    while step < loop_config['num_steps']:

        # Validation
        ce = 0.0
        acc = 0.0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x = preprocess(batch[0])
            y = batch[1]
            r = sess.run([m['ce'], m['acc']], feed_dict={
                m['x']: x, m['y_gt']: y, m['phase_train']: True
            })
            ce += r[0] / 100.0
            acc += r[1] / 100.0
        log.info('step {:d}, valid ce {:.4f}'.format(step, ce))
        ce_logger.add(step, ['', ce])
        acc_logger.add(step, ['', acc])

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = preprocess(batch[0])
            y = batch[1]
            st = time.time()
            r = sess.run([m['ce'], m['train_step']], feed_dict={
                m['x']: x, m['y_gt']: y, m['phase_train']: True
            })
            if step % 10 == 0:
                ce = r[0]
                step_time = (time.time() - st) * 1000
                log.info('{:d} ce {:.4f} t {:.2f}ms'.format(
                    step, ce, step_time))
                ce_logger.add(step, [ce, ''])
                acc_logger.add(step, [acc, ''])
                step_time_logger.add(step, step_time)

            step += 1

            # # Save model
            # if step % args.steps_per_ckpt == 0:
            #     saver.save_ckpt(exp_folder, sess, model_opt=opt,
            #                     global_step=step)

    sess.close()
