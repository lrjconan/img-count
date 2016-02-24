"""
This code implements Recurrent Instance Segmentation [1].

Author: Mengye Ren (m.ren@cs.toronto.edu)

Usage: python rec_ins_segm.py --help

Reference:
[1] B. Romera-Paredes, P. Torr. Recurrent Instance Segmentation. arXiv preprint
arXiv:1511.08250, 2015.
"""
import cslab_environ

import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time

from data_api import mnist
from utils import log_manager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger

import rec_ins_segm_models as models
import syncount_gen_data as data


def plot_samples(fname, x, y_out, s_out, y_gt, s_gt, match):
    """Plot some test samples."""
    num_row = y_out.shape[0]
    num_col = y_out.shape[1] + 1
    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    cmap = ['r', 'y', 'c', 'g', 'm', 'o']
    im_height = x.shape[1]
    im_with = x.shape[2]

    for ii in xrange(num_row):
        mnz = match[ii].nonzero()
        for jj in xrange(num_col):
            axarr[ii, jj].set_axis_off()
            if jj == 0:
                axarr[ii, jj].imshow(x[ii])
                for kk in xrange(y_gt.shape[1]):
                    nz = y_gt[ii, kk].nonzero()
                    if nz[0].size > 0:
                        top_left_x = nz[1].min()
                        top_left_y = nz[0].min()
                        bot_right_x = nz[1].max() + 1
                        bot_right_y = nz[0].max() + 1
                        axarr[ii, jj].add_patch(patches.Rectangle(
                            (top_left_x, top_left_y),
                            bot_right_x - top_left_x,
                            bot_right_y - top_left_y,
                            fill=False,
                            color=cmap[kk]))
                        axarr[ii, jj].add_patch(patches.Rectangle(
                            (top_left_x, top_left_y - 25),
                            25, 25,
                            fill=True,
                            color=cmap[kk]))
                        axarr[ii, jj].text(
                            top_left_x + 5, top_left_y - 5,
                            '{}'.format(kk), size=5)
            else:
                axarr[ii, jj].imshow(y_out[ii, jj - 1])
                matched = match[ii, jj - 1].nonzero()[0]
                axarr[ii, jj].text(0, 0, '{:.2f} {}'.format(
                    s_out[ii, jj - 1], matched),
                    color=(0, 0, 0), size=8)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=300)
    plt.clf()

    pass


def get_dataset(opt, num_train, num_valid):
    """Get train-valid split dataset for instance segmentation.

    Args:
        opt
        num_train
        num_valid
    Returns:
        dataset
            train
            valid
    """
    dataset = {}
    opt['num_examples'] = num_train
    raw_data = data.get_raw_data(opt, seed=2)
    image_data = data.get_image_data(opt, raw_data)
    segm_data = data.get_instance_segmentation_data(opt, image_data)
    dataset['train'] = segm_data

    opt['num_examples'] = num_valid
    raw_data = data.get_raw_data(opt, seed=3)
    image_data = data.get_image_data(opt, raw_data)
    segm_data = data.get_instance_segmentation_data(opt, image_data)
    dataset['valid'] = segm_data

    return dataset


def _get_batch_fn(dataset):
    """
    Preprocess mini-batch data given start and end indices.
    """
    def get_batch(idx):
        x_bat = dataset['input'][idx]
        y_bat = dataset['label_segmentation'][idx]
        s_bat = dataset['label_score'][idx]
        x_bat, y_bat, s_bat = preprocess(x_bat, y_bat, s_bat)

        return x_bat, y_bat, s_bat

    return get_batch


def preprocess(inp, label_segmentation, label_score):
    """Preprocess training data."""
    return (inp.astype('float32') / 255,
            label_segmentation.astype('float32'),
            label_score.astype('float32'))


def _parse_args():
    """Parse input arguments."""
    # Default dataset options
    kHeight = 224
    kWidth = 224
    kRadiusLower = 15
    kRadiusUpper = 45
    kBorderThickness = 3
    kNumExamples = 1000
    kMaxNumObjects = 6
    kNumObjectTypes = 1
    kSizeVar = 20
    kCenterVar = 20
    kOutputWindowSize = 128

    # Default model options
    # [224, 224,  3]
    # [112, 112,  4]
    # [56,  56,   8]
    # [28,  28,   8]
    # [14,  14,  12]
    # [7,   7,   16]
    # [7,   7,   12]
    # [7,   7,    6]
    # [14,  14, 8+12]
    # [28,  28,  6+8]
    # [56,  56,  4+8]
    # [112, 112, 4+4]
    # [224, 224, 2+3]
    # [224, 224,   1]

    kWeightDecay = 5e-5
    kBaseLearnRate = 1e-3
    kLearnRateDecay = 0.96
    kStepsPerDecay = 5000
    kLossMixRatio = 1.0
    kNumConv = 5
    kCnnFilterSize = [3, 3, 3, 3, 3]
    kCnnDepth = [4, 8, 8, 12, 16]
    kRnnType = 'conv_lstm'     # Choose from "conv_lstm", "lstm", and "gru"
    kConvLstmFilterSize = 3
    kConvLstmHiddenDepth = 12
    kRnnHiddenDim = 512

    kMlpDepth = 6
    kDcnnFilterSize = [3, 3, 3, 3, 3, 3]
    kDcnnDepth = [1, 2, 4, 4, 6, 8]
    kScoreMaxpool = 1

    # Default training options
    kNumSteps = 500000
    kStepsPerCkpt = 1000
    kStepsPerValid = 250

    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation')

    # Dataset options
    parser.add_argument('-height', default=kHeight, type=int,
                        help='Image height')
    parser.add_argument('-width', default=kWidth, type=int,
                        help='Image width')
    parser.add_argument('-radius_upper', default=kRadiusUpper, type=int,
                        help='Radius upper bound')
    parser.add_argument('-radius_lower', default=kRadiusLower, type=int,
                        help='Radius lower bound')
    parser.add_argument('-border_thickness', default=kBorderThickness,
                        type=int, help='Object border thickness')
    parser.add_argument('-num_ex', default=kNumExamples, type=int,
                        help='Number of examples')
    parser.add_argument('-max_num_objects', default=kMaxNumObjects, type=int,
                        help='Maximum number of objects')
    parser.add_argument('-num_object_types', default=kNumObjectTypes, type=int,
                        help='Number of object types')
    parser.add_argument('-center_var', default=kCenterVar, type=float,
                        help='Image patch center variance')
    parser.add_argument('-size_var', default=kSizeVar, type=float,
                        help='Image patch size variance')

    # Model options
    parser.add_argument('-model', default='original',
                        help='Which model to train')
    parser.add_argument('-weight_decay', default=kWeightDecay, type=float,
                        help='Weight L2 regularization')
    parser.add_argument('-base_learn_rate', default=kBaseLearnRate,
                        type=float, help='Model learning rate')
    parser.add_argument('-learn_rate_decay', default=kLearnRateDecay,
                        type=float, help='Model learning rate decay')
    parser.add_argument('-steps_per_decay', default=kStepsPerDecay,
                        type=int, help='Steps every learning rate decay')
    parser.add_argument('-loss_mix_ratio', default=kLossMixRatio, type=float,
                        help='Mix ratio between segmentation and score loss')
    parser.add_argument('-num_conv', default=kNumConv,
                        type=int, help='Number of convolutional layers')

    for ii in xrange(kNumConv):
        parser.add_argument('-cnn_{}_filter_size'.format(ii + 1),
                            default=kCnnFilterSize[ii], type=int,
                            help='CNN layer {} filter size'.format(ii + 1))
        parser.add_argument('-cnn_{}_depth'.format(ii + 1),
                            default=kCnnDepth[ii], type=int,
                            help='CNN layer {} depth'.format(ii + 1))

    for ii in xrange(kNumConv + 1):
        parser.add_argument('-dcnn_{}_filter_size'.format(ii),
                            default=kDcnnFilterSize[ii], type=int,
                            help='DCNN layer {} filter size'.format(ii))
        parser.add_argument('-dcnn_{}_depth'.format(ii),
                            default=kDcnnDepth[ii], type=int,
                            help='DCNN layer {} depth'.format(ii))

    parser.add_argument('-rnn_type', default=kRnnType, help='RNN type')
    parser.add_argument('-conv_lstm_filter_size', default=kConvLstmFilterSize,
                        type=int, help='Conv LSTM filter size')
    parser.add_argument('-conv_lstm_hid_depth', default=kConvLstmHiddenDepth,
                        type=int, help='Conv LSTM hidden depth')
    parser.add_argument('-rnn_hid_dim', default=kRnnHiddenDim,
                        type=int, help='RNN hidden dimension')
    parser.add_argument('-score_maxpool', default=kScoreMaxpool, type=int,
                        help='Max pooling ratio in the scoring function.')
    parser.add_argument('-mlp_depth', default=kMlpDepth,
                        type=int, help='MLP depth')

    # Extra model options (beta)
    parser.add_argument('-no_cum_min', action='store_true',
                        help='Whether cumulative minimum. Default yes.')
    parser.add_argument('-feed_output', action='store_true',
                        help='Whether to feed the output back to input.')
    parser.add_argument('-segm_loss_fn', default='iou',
                        help='Segmentation loss function, "iou" or "bce"')
    parser.add_argument('-use_deconv', action='store_true',
                        help='Whether to use deconvolution layer to upsample.')
    parser.add_argument('-use_bn', action='store_true',
                        help='Whether to use batch normalization.')
    parser.add_argument('-segm_dense_conn', action='store_true',
                        help='Whether to use dense connection to output segmentation.')
    parser.add_argument('-add_skip_conn', action='store_true',
                        help='Whether to add skip connection in the DCNN.')
    parser.add_argument('-score_use_core', action='store_true',
                        help='Use core MLP network to predict score.')

    # Training options
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-steps_per_valid', default=kStepsPerValid,
                        type=int, help='Number of steps per validation')
    parser.add_argument('-results', default='../results',
                        help='Model results folder')
    parser.add_argument('-logs', default='../results',
                        help='Training curve logs folder')
    parser.add_argument('-localhost', default='localhost',
                        help='Local domain name')
    parser.add_argument('-restore', default=None,
                        help='Model save folder to restore from')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    parser.add_argument('-num_samples_plot', default=10, type=int,
                        help='Number of samples to plot')

    args = parser.parse_args()

    return args


def get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None

    # Restore previously saved checkpoints.
    if args.restore:
        saver = Saver(args.restore)
        ckpt_info = saver.get_ckpt_info()
        model_opt = ckpt_info['model_opt']
        data_opt = ckpt_info['data_opt']
        ckpt_fname = ckpt_info['ckpt_fname']
        step = ckpt_info['step']
        model_id = ckpt_info['model_id']
        exp_folder = args.restore
    else:
        model_id = get_model_id('rec_ins_segm')

        cnn_filter_size_all = [args.cnn_1_filter_size,
                               args.cnn_2_filter_size,
                               args.cnn_3_filter_size,
                               args.cnn_4_filter_size,
                               args.cnn_5_filter_size]
        cnn_depth_all = [args.cnn_1_depth,
                         args.cnn_2_depth,
                         args.cnn_3_depth,
                         args.cnn_4_depth,
                         args.cnn_5_depth]
        dcnn_filter_size_all = [args.dcnn_0_filter_size,
                                args.dcnn_1_filter_size,
                                args.dcnn_2_filter_size,
                                args.dcnn_3_filter_size,
                                args.dcnn_4_filter_size,
                                args.dcnn_5_filter_size]
        dcnn_depth_all = [args.dcnn_0_depth,
                          args.dcnn_1_depth,
                          args.dcnn_2_depth,
                          args.dcnn_3_depth,
                          args.dcnn_4_depth,
                          args.dcnn_5_depth]
        model_opt = {
            'inp_height': args.height,
            'inp_width': args.width,
            'inp_depth': 3,
            'timespan': args.max_num_objects + 1,
            'weight_decay': args.weight_decay,
            'base_learn_rate': args.base_learn_rate,
            'learn_rate_decay': args.learn_rate_decay,
            'steps_per_decay': args.steps_per_decay,
            'loss_mix_ratio': args.loss_mix_ratio,
            'cnn_filter_size': cnn_filter_size_all[: args.num_conv],
            'cnn_depth': cnn_depth_all[: args.num_conv],
            'dcnn_filter_size': dcnn_filter_size_all[: args.num_conv + 1][::-1],
            'dcnn_depth': dcnn_depth_all[: args.num_conv + 1][::-1],
            'rnn_type': args.rnn_type,
            'conv_lstm_filter_size': args.conv_lstm_filter_size,
            'conv_lstm_hid_depth': args.conv_lstm_hid_depth,
            'rnn_hid_dim': args.rnn_hid_dim,
            'mlp_depth': args.mlp_depth,
            'score_maxpool': args.score_maxpool,

            # Test arguments
            'cum_min': not args.no_cum_min,
            'feed_output': args.feed_output,
            'segm_loss_fn': args.segm_loss_fn,
            'use_deconv': args.use_deconv,
            'use_bn': args.use_bn,
            'segm_dense_conn': args.segm_dense_conn,
            'add_skip_conn': args.add_skip_conn,
            'score_use_core': args.score_use_core
        }
        data_opt = {
            'height': args.height,
            'width': args.width,
            'radius_upper': args.radius_upper,
            'radius_lower': args.radius_lower,
            'border_thickness': args.border_thickness,
            'max_num_objects': args.max_num_objects,
            'num_object_types': args.num_object_types,
            'center_var': args.center_var,
            'size_var': args.size_var
        }
        step = 0
        exp_folder = os.path.join(args.results, model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    # Logger
    if args.logs:
        logs_folder = args.logs
        logs_folder = os.path.join(logs_folder, model_id)
        log = logger.get(os.path.join(logs_folder, 'raw'))
    else:
        log = logger.get()

    # Log arguments
    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    # Train loop options
    train_opt = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt,
        'steps_per_valid': args.steps_per_valid
    }

    dataset = get_dataset(data_opt, args.num_ex, args.num_ex / 10)
    m = models.get_model(args.model, model_opt, device=device, train=True)
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series logger
    if args.logs:
        train_loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'train_loss.csv'), 'train loss',
            name='Training loss',
            buffer_size=10)
        valid_loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'valid_loss.csv'), 'valid loss',
            name='Validation loss',
            buffer_size=1)
        valid_iou_hard_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'valid_iou_hard.csv'), 'valid iou',
            name='Validation IoU hard',
            buffer_size=1)
        valid_iou_soft_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'valid_iou_soft.csv'), 'valid iou',
            name='Validation IoU soft',
            buffer_size=1)
        valid_count_acc_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'valid_count_acc.csv'),
            'valid count acc',
            name='Validation count accuracy',
            buffer_size=1)
        learn_rate_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'learn_rate.csv'),
            'learning rate',
            name='Learning rate',
            buffer_size=10)
        step_time_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
            name='Step time',
            buffer_size=10)

        log_manager.register(log.filename, 'plain', 'Raw logs')

        model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
        saver.save_opt(model_opt_fname, model_opt)
        log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

        valid_sample_img = LazyRegisterer(os.path.join(
            logs_folder, 'valid_sample_img.png'),
            'image', 'Validation samples')
        log.info(
            ('Visualization can be viewed at: '
             'http://{}/deep-dashboard?id={}').format(
                args.localhost, model_id))

    num_ex_train = dataset['train']['input'].shape[0]
    num_ex_valid = dataset['valid']['input'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    get_batch_valid = _get_batch_fn(dataset['valid'])
    batch_size_train = 32
    batch_size_valid = 32
    log.info('Number of validation examples: {}'.format(num_ex_valid))
    log.info('Validation batch size: {}'.format(batch_size_valid))
    log.info('Number of training examples: {}'.format(num_ex_train))
    log.info('Training batch size: {}'.format(batch_size_train))

    def run_validation():
        # Validation
        loss = 0.0
        iou_hard = 0.0
        iou_soft = 0.0
        count_acc = 0.0
        segm_loss = 0.0
        conf_loss = 0.0
        log.info('Running validation')
        for _x, _y, _s in BatchIterator(num_ex_valid,
                                        batch_size=batch_size_valid,
                                        get_fn=get_batch_valid,
                                        progress_bar=False):
            results = sess.run([m['loss'], m['segm_loss'], m['conf_loss'],
                                m['iou_soft'], m['iou_hard'], m['count_acc']],
                               feed_dict={
                m['x']: _x,
                m['phase_train']: False,
                m['y_gt']: _y,
                m['s_gt']: _s
            })
            _loss = results[0]
            _segm_loss = results[1]
            _conf_loss = results[2]
            _iou_soft = results[3]
            _iou_hard = results[4]
            _count_acc = results[5]
            loss += _loss * batch_size_valid / float(num_ex_valid)
            segm_loss += _segm_loss * batch_size_valid / float(num_ex_valid)
            conf_loss += _conf_loss * batch_size_valid / float(num_ex_valid)
            iou_soft += _iou_soft * batch_size_valid / float(num_ex_valid)
            iou_hard += _iou_hard * batch_size_valid / float(num_ex_valid)
            count_acc += _count_acc * batch_size_valid / float(num_ex_valid)

        log.info(('{:d} valid loss {:.4f} segm_loss {:.4f} conf_loss {:.4f} '
                  'iou soft {:.4f} iou hard {:.4f} count acc {:.4f}').format(
            step, loss, segm_loss, conf_loss, iou_soft, iou_hard, count_acc))

        if args.logs:
            _x, _y, _s = get_batch_valid(np.arange(args.num_samples_plot))
            _y_out, _s_out, _match = sess.run(
                [m['y_out'], m['s_out'], m['match']], feed_dict={
                    m['x']: _x,
                    m['phase_train']: False,
                    m['y_gt']: _y,
                    m['s_gt']: _s
                })
            plot_samples(valid_sample_img.get_fname(), _x, _y_out, _s_out, _y,
                         _s, _match)
            valid_loss_logger.add(step, loss)
            valid_iou_soft_logger.add(step, iou_soft)
            valid_iou_hard_logger.add(step, iou_hard)
            valid_count_acc_logger.add(step, count_acc)
            if not valid_sample_img.is_registered():
                valid_sample_img.register()

        pass

    # Train loop
    for x_bat, y_bat, s_bat in BatchIterator(num_ex_train,
                                             batch_size=batch_size_train,
                                             get_fn=get_batch_train,
                                             cycle=True,
                                             progress_bar=False):
        # Run validation
        if step % train_opt['steps_per_valid'] == 0:
            run_validation()

        # Train step
        start_time = time.time()
        r = sess.run([m['loss'], m['segm_loss'], m['conf_loss'], m['train_step']], feed_dict={
            m['x']: x_bat,
            m['phase_train']: True,
            m['y_gt']: y_bat,
            m['s_gt']: s_bat
        })

        # Print statistics
        if step % 5 == 0:
            step_time = (time.time() - start_time) * 1000
            loss = r[0]
            log.info('{:d} train loss {:.4f} {:.4f} {:.4f} t {:.2f}ms'.format(
                step, loss, r[1], r[2], step_time))

            if args.logs:
                train_loss_logger.add(step, loss)
                learn_rate_logger.add(step, sess.run(m['learn_rate']))
                step_time_logger.add(step, step_time)

        # Model ID reminder
        if step % 100 == 0:
            log.info('model id {}'.format(model_id))

        # Save model
        if step % train_opt['steps_per_ckpt'] == 0:
            saver.save(sess, global_step=step)

        step += 1

        # Termination
        if step > train_opt['num_steps']:
            break

    sess.close()
    train_loss_logger.close()
    valid_loss_logger.close()
    valid_iou_soft_logger.close()
    valid_iou_hard_logger.close()
    valid_count_acc_logger.close()
    step_time_logger.close()

    pass
