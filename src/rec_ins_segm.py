"""
This code implements Recurrent Instance Segmentation [1].

Author: Mengye Ren (m.ren@cs.toronto.edu)

Usage: python rec_ins_segm.py --help

Reference:
[1] B. Romera-Paredes, P. Torr. Recurrent Instance Segmentation. arXiv preprint
arXiv:1511.08250, 2015.
"""
from __future__ import division

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

from data_api import synth_shape
from data_api import cvppp

from utils import log_manager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger

import rec_ins_segm_models as models


def plot_samples(fname, x_orig, x, y_out, s_out, y_gt, s_gt, match):
    """Plot some test samples."""
    num_ex = y_out.shape[0]
    offset = 2
    num_items = y_out.shape[1] + offset
    max_items_per_row = 9
    num_rows_per_ex = int(np.ceil(num_items / max_items_per_row))
    if num_items > max_items_per_row:
        num_col = max_items_per_row
        num_row = num_rows_per_ex * num_ex
    else:
        num_row = num_ex
        num_col = num_items

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    cmap = ['r', 'y', 'c', 'g', 'm']
    gradient = np.linspace(0, 1, 256)
    im_height = x.shape[1]
    im_with = x.shape[2]

    for row in xrange(num_row):
        for col in xrange(num_col):
            axarr[row, col].set_axis_off()

    for ii in xrange(num_ex):
        mnz = match[ii].nonzero()
        for jj in xrange(num_items):
            col = jj % max_items_per_row
            row = num_rows_per_ex * ii + int(jj / max_items_per_row)
            if jj == 0:
                axarr[row, col].imshow(x_orig[ii])
            elif jj == 1:
                axarr[row, col].imshow(x[ii])
                for kk in xrange(y_gt.shape[1]):
                    nz = y_gt[ii, kk].nonzero()
                    if nz[0].size > 0:
                        top_left_x = nz[1].min()
                        top_left_y = nz[0].min()
                        bot_right_x = nz[1].max() + 1
                        bot_right_y = nz[0].max() + 1
                        axarr[row, col].add_patch(patches.Rectangle(
                            (top_left_x, top_left_y),
                            bot_right_x - top_left_x,
                            bot_right_y - top_left_y,
                            fill=False,
                            color=cmap[kk % len(cmap)]))
                        axarr[row, col].add_patch(patches.Rectangle(
                            (top_left_x, top_left_y - 25),
                            25, 25,
                            fill=True,
                            color=cmap[kk % len(cmap)]))
                        axarr[row, col].text(
                            top_left_x + 5, top_left_y - 5,
                            '{}'.format(kk), size=5)
            else:
                axarr[row, col].imshow(y_out[ii, jj - offset])
                matched = match[ii, jj - offset].nonzero()[0]
                axarr[row, col].text(0, 0, '{:.2f} {}'.format(
                    s_out[ii, jj - offset], matched),
                    color=(0, 0, 0), size=8)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=300)
    plt.close('all')

    pass


def get_dataset(dataset_name, opt, num_train, num_valid):
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
    if dataset_name == 'synth_shape':
        opt['num_examples'] = num_train
        dataset['train'] = synth_shape.get_dataset(opt, seed=2)
        opt['num_examples'] = num_valid
        dataset['valid'] = synth_shape.get_dataset(opt, seed=3)
    elif dataset_name == 'cvppp':
        if os.path.exists('/u/mren'):
            dataset_folder = '/ais/gobi3/u/mren/data/lsc/A1'
        else:
            dataset_folder = '/home/mren/data/LSCData/A1'
        _all_data = cvppp.get_dataset(dataset_folder, opt)
        split = 103
        random = np.random.RandomState(2)
        idx = np.arange(_all_data['input'].shape[0])
        random.shuffle(idx)
        train_idx = idx[: split]
        valid_idx = idx[split:]
        log.info('Train index: {}'.format(train_idx))
        log.info('Valid index: {}'.format(valid_idx))
        dataset['train'] = {
            'input': _all_data['input'][train_idx],
            'label_segmentation': _all_data['label_segmentation'][train_idx],
            'label_score': _all_data['label_score'][train_idx]
        }
        dataset['valid'] = {
            'input': _all_data['input'][valid_idx],
            'label_segmentation': _all_data['label_segmentation'][valid_idx],
            'label_score': _all_data['label_score'][valid_idx]
        }

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
    kDataset = 'synth_shape'
    kHeight = 224
    kWidth = 224
    kPadding = 16
    # (Below are only valid options for synth_shape dataset)
    kRadiusLower = 15
    kRadiusUpper = 45
    kBorderThickness = 3
    kNumExamples = 1000
    kMaxNumObjects = 6
    kNumObjectTypes = 1
    kSizeVar = 20
    kCenterVar = 20

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
    kStepsPerLearnRateDecay = 5000
    kStepsPerLog = 10
    kLossMixRatio = 1.0
    kNumConv = 5
    kCnnFilterSize = [3, 3, 3, 3, 3]
    kCnnDepth = [4, 8, 8, 12, 16]
    kRnnType = 'lstm'                  # {"conv_lstm", "lstm", "gru"}
    kConvLstmFilterSize = 3
    kConvLstmHiddenDepth = 12
    kRnnHiddenDim = 512

    kNumMlpLayers = 2
    kMlpDepth = 6
    kMlpDropout = 0.5
    kDcnnFilterSize = [3, 3, 3, 3, 3, 3]
    kDcnnDepth = [1, 2, 4, 4, 6, 8]
    kScoreMaxpool = 1

    # Default training options
    kNumSteps = 500000
    kStepsPerCkpt = 1000
    kStepsPerValid = 250
    kStepsPerPlot = 50
    kBatchSize = 32

    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation')

    # Dataset options
    parser.add_argument('-dataset', default=kDataset,
                        help='Name of the dataset')
    parser.add_argument('-height', default=kHeight, type=int,
                        help='Image height')
    parser.add_argument('-width', default=kWidth, type=int,
                        help='Image width')
    parser.add_argument('-padding', default=kPadding, type=int,
                        help='Apply additional padding for random cropping')

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
    parser.add_argument('-steps_per_learn_rate_decay',
                        default=kStepsPerLearnRateDecay,
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
    parser.add_argument('-num_mlp_layers', default=kNumMlpLayers,
                        type=int, help='Number of MLP layers')
    parser.add_argument('-mlp_depth', default=kMlpDepth,
                        type=int, help='MLP depth')
    parser.add_argument('-mlp_dropout', default=kMlpDropout,
                        type=float, help='MLP dropout')

    # Extra model options (beta)
    parser.add_argument('-no_cum_min', action='store_true',
                        help='Whether cumulative minimum. Default yes.')
    # parser.add_argument('-feed_output', action='store_true',
    #                     help='Whether to feed the output back to input.')
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
    parser.add_argument('-steps_per_plot', default=kStepsPerPlot,
                        type=int, help='Number of steps per plot')
    parser.add_argument('-steps_per_log', default=kStepsPerLog,
                        type=int, help='Number of steps per log')
    parser.add_argument('-batch_size', default=kBatchSize,
                        type=int, help='Size of a mini-batch')
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
    parser.add_argument('-save_ckpt', action='store_true',
                        help='Whether to store checkpoints')

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

        if args.dataset == 'synth_shape':
            timespan = args.max_num_objects + 1
        elif args.dataset == 'cvppp':
            timespan = 21
        else:
            raise Exception('Unknown dataset name')

        model_opt = {
            'inp_height': args.height,
            'inp_width': args.width,
            'inp_depth': 3,
            'padding': args.padding,
            'timespan': timespan,
            'weight_decay': args.weight_decay,
            'base_learn_rate': args.base_learn_rate,
            'learn_rate_decay': args.learn_rate_decay,
            'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
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
            'num_mlp_layers': args.num_mlp_layers,
            'mlp_dropout': args.mlp_dropout,

            # Test arguments
            'cum_min': not args.no_cum_min,
            # 'feed_output': args.feed_output,
            'segm_loss_fn': args.segm_loss_fn,
            # 'use_deconv': args.use_deconv,
            'use_deconv': True,
            'use_bn': args.use_bn,
            # 'segm_dense_conn': args.segm_dense_conn,
            'segm_dense_conn': True,
            # 'add_skip_conn': args.add_skip_conn,
            'add_skip_conn': True,
            # 'score_use_core': args.score_use_core
            'score_use_core': True
        }
        data_opt = {
            'height': args.height,
            'width': args.width,
            'padding': args.padding,
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

    if not args.save_ckpt:
        log.warning(
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

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
        'steps_per_valid': args.steps_per_valid,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log
    }

    log.info('Building model')
    m = models.get_model(args.model, model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(args.dataset, data_opt,
                          args.num_ex, int(args.num_ex / 10))

    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series logger
    if args.logs:
        loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
            name='Loss',
            buffer_size=1)
        iou_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'iou.csv'),
            ['train soft', 'valid soft', 'train hard', 'valid hard'],
            name='IoU',
            buffer_size=1)
        count_acc_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'count_acc.csv'),
            ['train', 'valid'],
            name='Count accuracy',
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

        num_cnn = len(model_opt['cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])
        cnn_bn_loggers = []
        for ii in xrange(num_cnn):
            _cnn_bn_logger = TimeSeriesLogger(
                os.path.join(logs_folder, 'cnn_{}_bn.csv'.format(ii)),
                ['train batch mean', 'valid batch mean', 'train batch variance',
                    'valid batch variance', 'ema mean', 'ema variance'],
                name='CNN {} batch norm stats'.format(ii),
                buffer_size=1)
            cnn_bn_loggers.append(_cnn_bn_logger)

        dcnn_bn_loggers = []
        for ii in xrange(num_dcnn):
            _dcnn_bn_logger = TimeSeriesLogger(
                os.path.join(logs_folder, 'dcnn_{}_bn.csv'.format(ii)),
                ['train batch mean', 'valid batch mean', 'train batch variance',
                    'valid batch variance', 'ema mean', 'ema variance'],
                name='D-CNN {} batch norm stats'.format(ii),
                buffer_size=1)
            dcnn_bn_loggers.append(_dcnn_bn_logger)

        log_manager.register(log.filename, 'plain', 'Raw logs')

        model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
        saver.save_opt(model_opt_fname, model_opt)
        log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

        valid_sample_img = LazyRegisterer(os.path.join(
            logs_folder, 'valid_sample_img.png'),
            'image', 'Validation samples')
        train_sample_img = LazyRegisterer(os.path.join(
            logs_folder, 'train_sample_img.png'),
            'image', 'Training samples')
        log.info(
            ('Visualization can be viewed at: '
             'http://{}/deep-dashboard?id={}').format(
                args.localhost, model_id))

    num_ex_train = dataset['train']['input'].shape[0]
    num_ex_valid = dataset['valid']['input'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    get_batch_valid = _get_batch_fn(dataset['valid'])
    batch_size = args.batch_size
    log.info('Number of validation examples: {}'.format(num_ex_valid))
    log.info('Number of training examples: {}'.format(num_ex_train))
    log.info('Batch size: {}'.format(batch_size))

    def run_samples():
        """Samples"""
        def _run_samples(x, y, s, phase_train, fname):
            x2, y2, y_out, s_out, match = sess.run(
                [m['x_trans'], m['y_gt_trans'], m['y_out'], m['s_out'],
                 m['match']],
                feed_dict={
                    m['x']: x,
                    m['phase_train']: phase_train,
                    m['y_gt']: y,
                    m['s_gt']: s
                })

            plot_samples(fname, x_orig=x, x=x2, y_out=y_out, s_out=s_out,
                         y_gt=y2, s_gt=s, match=match)
            
        if args.logs:
            # Plot some samples.
            log.info('Plotting validation samples')
            _x, _y, _s = get_batch_valid(np.arange(args.num_samples_plot))
            _x, _y, _s = get_batch_valid(np.arange(args.num_samples_plot))
            _run_samples(_x, _y, _s, False, valid_sample_img.get_fname())
            if not valid_sample_img.is_registered():
                valid_sample_img.register()

            log.info('Plotting training samples')
            _x, _y, _s = get_batch_train(np.arange(args.num_samples_plot))
            _run_samples(_x, _y, _s, True, train_sample_img.get_fname())
            if not train_sample_img.is_registered():
                train_sample_img.register()

        pass

    def run_validation():
        # Validation
        loss = 0.0
        iou_hard = 0.0
        iou_soft = 0.0
        count_acc = 0.0
        segm_loss = 0.0
        conf_loss = 0.0
        num_cnn = len(model_opt['cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])
        cnn_bm = [0.0] * num_cnn
        cnn_bv = [0.0] * num_cnn
        cnn_em = [0.0] * num_cnn
        cnn_ev = [0.0] * num_cnn
        dcnn_bm = [0.0] * num_dcnn
        dcnn_bv = [0.0] * num_dcnn
        dcnn_em = [0.0] * num_dcnn
        dcnn_ev = [0.0] * num_dcnn
        log.info('Running validation')
        for _x, _y, _s in BatchIterator(num_ex_valid,
                                        batch_size=batch_size,
                                        get_fn=get_batch_valid,
                                        progress_bar=False):
            results_list = [m['loss'], m['segm_loss'], m['conf_loss'],
                            m['iou_soft'], m['iou_hard'], m['count_acc']]
            offset = len(results_list)
            for ii in xrange(num_cnn):
                results_list.append(m['cnn_{}_bm'.format(ii)])
                results_list.append(m['cnn_{}_bv'.format(ii)])
                results_list.append(m['cnn_{}_em'.format(ii)])
                results_list.append(m['cnn_{}_ev'.format(ii)])

            for ii in xrange(num_dcnn):
                results_list.append(m['dcnn_{}_bm'.format(ii)])
                results_list.append(m['dcnn_{}_bv'.format(ii)])
                results_list.append(m['dcnn_{}_em'.format(ii)])
                results_list.append(m['dcnn_{}_ev'.format(ii)])

            results = sess.run(results_list,
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

            for ii in xrange(num_cnn):
                _cnn_bm = results[offset]
                _cnn_bv = results[offset + 1]
                _cnn_em = results[offset + 2]
                _cnn_ev = results[offset + 3]
                offset += 4

            for ii in xrange(num_dcnn):
                _dcnn_bm = results[offset]
                _dcnn_bv = results[offset + 1]
                _dcnn_em = results[offset + 2]
                _dcnn_ev = results[offset + 3]
                offset += 4

            num_ex_batch = _x.shape[0]
            loss += _loss * num_ex_batch / num_ex_valid
            segm_loss += _segm_loss * num_ex_batch / num_ex_valid
            conf_loss += _conf_loss * num_ex_batch / num_ex_valid
            iou_soft += _iou_soft * num_ex_batch / num_ex_valid
            iou_hard += _iou_hard * num_ex_batch / num_ex_valid
            count_acc += _count_acc * num_ex_batch / num_ex_valid
            for ii in xrange(num_cnn):
                cnn_bm[ii] += _cnn_bm * num_ex_batch / num_ex_valid
                cnn_bv[ii] += _cnn_bv * num_ex_batch / num_ex_valid
                cnn_em[ii] += _cnn_em * num_ex_batch / num_ex_valid
                cnn_ev[ii] += _cnn_ev * num_ex_batch / num_ex_valid
            for ii in xrange(num_dcnn):
                dcnn_bm[ii] += _dcnn_bm * num_ex_batch / num_ex_valid
                dcnn_bv[ii] += _dcnn_bv * num_ex_batch / num_ex_valid
                dcnn_em[ii] += _dcnn_em * num_ex_batch / num_ex_valid
                dcnn_ev[ii] += _dcnn_ev * num_ex_batch / num_ex_valid

        log.info(('{:d} valid loss {:.4f} segm_loss {:.4f} conf_loss {:.4f} '
                  'iou soft {:.4f} iou hard {:.4f} count acc {:.4f}').format(
            step, loss, segm_loss, conf_loss, iou_soft, iou_hard, count_acc))

        if args.logs:
            loss_logger.add(step, ['', loss])
            iou_logger.add(step, ['', iou_soft, '', iou_hard])
            count_acc_logger.add(step, ['', count_acc])
            for ii in xrange(num_cnn):
                cnn_bn_loggers[ii].add(
                    step, ['', cnn_bm[ii], '', cnn_bv[ii], '', ''])
            for ii in xrange(num_dcnn):
                dcnn_bn_loggers[ii].add(
                    step, ['', dcnn_bm[ii], '', dcnn_bv[ii], '', ''])

        pass

    def train_step(step, x, y, s):
        """Train step"""
        start_time = time.time()
        num_cnn = len(model_opt['cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])
        cnn_bm = [0.0] * num_cnn
        cnn_bv = [0.0] * num_cnn
        cnn_em = [0.0] * num_cnn
        cnn_ev = [0.0] * num_cnn
        dcnn_bm = [0.0] * num_dcnn
        dcnn_bv = [0.0] * num_dcnn
        dcnn_em = [0.0] * num_dcnn
        dcnn_ev = [0.0] * num_dcnn

        results_list = [m['loss'], m['segm_loss'], m['conf_loss'],
                        m['iou_soft'], m['iou_hard'], m['count_acc'],
                        m['learn_rate']]
        offset = len(results_list)

        for ii in xrange(num_cnn):
            results_list.append(m['cnn_{}_bm'.format(ii)])
            results_list.append(m['cnn_{}_bv'.format(ii)])
            results_list.append(m['cnn_{}_em'.format(ii)])
            results_list.append(m['cnn_{}_ev'.format(ii)])

        for ii in xrange(num_dcnn):
            results_list.append(m['dcnn_{}_bm'.format(ii)])
            results_list.append(m['dcnn_{}_bv'.format(ii)])
            results_list.append(m['dcnn_{}_em'.format(ii)])
            results_list.append(m['dcnn_{}_ev'.format(ii)])

        results_list.append(m['train_step'])

        results = sess.run(results_list, feed_dict={
            m['x']: x_bat,
            m['phase_train']: True,
            m['y_gt']: y_bat,
            m['s_gt']: s_bat
        })

        # Print statistics
        if step % train_opt['steps_per_log'] == 0:
            loss = results[0]
            segm_loss = results[1]
            conf_loss = results[2]
            iou_soft = results[3]
            iou_hard = results[4]
            count_acc = results[5]
            learn_rate = results[6]

            for ii in xrange(num_cnn):
                cnn_bm[ii] = results[offset]
                cnn_bv[ii] = results[offset + 1]
                cnn_em[ii] = results[offset + 2]
                cnn_ev[ii] = results[offset + 3]
                offset += 4

            for ii in xrange(num_dcnn):
                dcnn_bm[ii] = results[offset]
                dcnn_bv[ii] = results[offset + 1]
                dcnn_em[ii] = results[offset + 2]
                dcnn_ev[ii] = results[offset + 3]
                offset += 4

            step_time = (time.time() - start_time) * 1000
            log.info('{:d} train loss {:.4f} {:.4f} {:.4f} t {:.2f}ms'.format(
                step, loss, segm_loss, conf_loss, step_time))

            if args.logs:
                loss_logger.add(step, [loss, ''])
                iou_logger.add(step, [iou_soft, '', iou_hard, ''])
                count_acc_logger.add(step, [count_acc, ''])
                learn_rate_logger.add(step, learn_rate)
                step_time_logger.add(step, step_time)
                for ii in xrange(num_cnn):
                    cnn_bn_loggers[ii].add(
                        step, [cnn_bm[ii], '', cnn_bv[ii], '', cnn_em[ii],
                               cnn_ev[ii]])
                for ii in xrange(num_dcnn):
                    dcnn_bn_loggers[ii].add(
                        step, [dcnn_bm[ii], '', dcnn_bv[ii], '', dcnn_em[ii],
                               dcnn_ev[ii]])
        pass

    def train_loop(step=0):
        # Train loop
        for x_bat, y_bat, s_bat in BatchIterator(num_ex_train,
                                                 batch_size=batch_size,
                                                 get_fn=get_batch_train,
                                                 cycle=True,
                                                 progress_bar=False):
            # Run validation
            if step % train_opt['steps_per_valid'] == 0:
                run_validation()

            if step % train_opt['steps_per_plot'] == 0:
                run_samples()

            # Train step
            train_step(step, x_bat, y_bat, s_bat)

            # Model ID reminder
            if step % (10 * train_opt['steps_per_log']) == 0:
                log.info('model id {}'.format(model_id))

            # Save model
            if args.save_ckpt and step % train_opt['steps_per_ckpt'] == 0:
                saver.save(sess, global_step=step)

            step += 1

            # Termination
            if step > train_opt['num_steps']:
                break
        pass

    train_loop(step=step)

    sess.close()
    loss_logger.close()
    iou_logger.close()
    count_acc_logger.close()
    learn_rate_logger.close()
    step_time_logger.close()

    pass
