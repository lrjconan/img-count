
"""
Recurrent instance segmentation with attention.

Usage: python rec_ins_segm_attn.py --help
"""
from __future__ import division

import cslab_environ

import argparse
import datetime
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import sys
import tensorflow as tf
import time

from data_api import synth_shape
from data_api import cvppp
from data_api import kitti

from utils import logger
from utils import plot_utils as pu
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.log_manager import LogManager
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger

import ris_base as base
import ris_vanilla_model as vanilla_model
# import ris_attn_model_old as attention_model
import ris_attn_model as attention_model
import ris_double_attn_model as double_attention_model

log = logger.get()

kSynthShapeInpHeight = 224
kSynthShapeInpWidth = 224
kCvpppInpHeight = 224
kCvpppInpWidth = 224
kCvpppNumObj = 20
kKittiInpHeight = 128
kKittiInpWidth = 448
kKittiNumObj = 19


def get_model(opt, device='/cpu:0'):
    """Model router."""
    name = opt['type']
    if name == 'vanilla':
        return vanilla_model.get_model(opt, device=device)
    elif name == 'attention':
        return attention_model.get_model(opt, device=device)
    elif name == 'double_attention':
        return double_attention_model.get_model(opt, device=device)
    else:
        raise Exception('Unknown model name "{}"'.format(name))

    pass


def plot_double_attention(fname, x, glimpse_map, max_items_per_row=9):
    """Plot double attention.

    Args:
        fname: str, image output filename.
        x: [B, H, W, 3], input image.
        glimpse_map: [B, T, T2, H', W']: glimpse attention map.
    """
    num_ex = y_out.shape[0]
    timespan = glimpse_map.shape[1]
    num_glimpse = glimpse_map.shape[2]
    num_items = timespan * num_glimpse
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)


def plot_total_instances(fname, y_out, s_out, max_items_per_row=9):
    """Plot cumulative image with different colour at each timestep.

    Args:
        y_out: [B, T, H, W]
    """
    num_ex = y_out.shape[0]
    num_items = y_out.shape[1]
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)

    cmap2 = np.array([[192, 57, 43],
                      [243, 156, 18],
                      [26, 188, 156],
                      [41, 128, 185],
                      [142, 68, 173],
                      [44, 62, 80],
                      [127, 140, 141],
                      [17, 75, 95],
                      [2, 128, 144],
                      [228, 253, 225],
                      [69, 105, 144],
                      [244, 91, 105],
                      [91, 192, 235],
                      [253, 231, 76],
                      [155, 197, 61],
                      [229, 89, 52],
                      [250, 121, 33]], dtype='uint8')

    for ii in xrange(num_ex):
        total_img = np.zeros([y_out.shape[2], y_out.shape[3], 3])
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            if s_out[ii, jj] > 0.5:
                total_img += np.expand_dims(
                    (y_out[ii, jj] > 0.5).astype('uint8'), 2) * \
                    cmap2[jj % cmap2.shape[0]]
            axarr[row, col].imshow(total_img)
            total_img = np.copy(total_img)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass


def plot_input(fname, x, y_gt, s_gt, max_items_per_row=9):
    """Plot input, transformed input and output groundtruth sequence.
    """
    num_ex = y_gt.shape[0]
    num_items = y_gt.shape[1]
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    pu.set_axis_off(axarr, num_row, num_col)
    cmap = ['r', 'y', 'c', 'g', 'm']

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            axarr[row, col].imshow(x[ii])
            nz = y_gt[ii, jj].nonzero()
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
                    color=cmap[jj % len(cmap)]))
                axarr[row, col].add_patch(patches.Rectangle(
                    (top_left_x, top_left_y - 25),
                    25, 25,
                    fill=True,
                    color=cmap[jj % len(cmap)]))
                axarr[row, col].text(
                    top_left_x + 5, top_left_y - 5,
                    '{}'.format(jj), size=5)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass


def plot_output(fname, y_out, s_out, match, attn=None, max_items_per_row=9):
    """Plot some test samples.

    Args:
        fname: str, image output filename.
        x: [B, H, W, D], image after input transformation.
        y_out: [B, T, H, W, D], segmentation output of the model.
        s_out: [B, T], confidence score output of the model.
        match: [B, T, T], matching matrix.
        attn: ([B, T, 2], [B, T, 2]), top left and bottom right coordinates of
        the attention box.
    """
    num_ex = y_out.shape[0]
    num_items = y_out.shape[1]
    num_row, num_col, calc = pu.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    cmap = ['r', 'y', 'c', 'g', 'm']

    if attn:
        attn_top_left_y = attn[0][:, :, 0]
        attn_top_left_x = attn[0][:, :, 1]
        attn_bot_right_y = attn[1][:, :, 0]
        attn_bot_right_x = attn[1][:, :, 1]

    pu.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            axarr[row, col].imshow(y_out[ii, jj])
            matched = match[ii, jj].nonzero()[0]
            axarr[row, col].text(0, 0, '{:.2f} {}'.format(
                s_out[ii, jj], matched),
                color=(0, 0, 0), size=8)

            if attn:
                # Plot attention box.
                axarr[row, col].add_patch(patches.Rectangle(
                    (attn_top_left_x[ii, jj], attn_top_left_y[ii, jj]),
                    attn_bot_right_x[ii, jj] - attn_top_left_x[ii, jj],
                    attn_bot_right_y[ii, jj] - attn_top_left_y[ii, jj],
                    fill=False,
                    color='g'))

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass


def get_dataset(dataset_name, opt):
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
        opt['num_examples'] = opt['num_train']
        dataset['train'] = synth_shape.get_dataset(opt, seed=2)
        opt['num_examples'] = opt['num_valid']
        dataset['valid'] = synth_shape.get_dataset(opt, seed=3)
    elif dataset_name == 'cvppp':
        if os.path.exists('/u/mren'):
            dataset_folder = '/ais/gobi3/u/mren/data/lsc/A1'
        else:
            dataset_folder = '/home/mren/data/LSCData/A1'
        _all_data = cvppp.get_dataset(dataset_folder, opt)

        if opt['has_valid']:
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
        else:
            random = np.random.RandomState(2)
            idx = np.arange(_all_data['input'].shape[0])
            random.shuffle(idx)
            dataset['train'] = {
                'input': _all_data['input'][idx],
                'label_segmentation': _all_data['label_segmentation'][idx],
                'label_score': _all_data['label_score'][idx]
            }
    elif dataset_name == 'kitti':
        if os.path.exists('/u/mren'):
            dataset_folder = '/ais/gobi3/u/mren/data/kitti/object'
        else:
            dataset_folder = '/home/mren/data/kitti'
        opt['timespan'] = 20
        opt['num_examples'] = opt['num_train']
        dataset['train'] = kitti.get_dataset(
            dataset_folder, opt, split='train')
        opt['num_examples'] = opt['num_valid']
        dataset['valid'] = kitti.get_dataset(
            dataset_folder, opt, split='valid')
    else:
        raise Exception('Unknown dataset name')

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


def _run_model(m, names, feed_dict):
    symbol_list = [m[r] for r in names]
    results = sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, names):
        results_dict[name] = rr

    return results_dict


def preprocess(inp, label_segmentation, label_score):
    """Preprocess training data."""
    return (inp.astype('float32') / 255,
            label_segmentation.astype('float32'),
            label_score.astype('float32'))


def _add_dataset_args(parser):
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

    pass


def _add_model_args(parser):
    # Original model default options
    kCnnFilterSize = '3,3,3,3,3'
    kCnnDepth = '4,8,8,12,16'
    kCnnPool = '2,2,2,2,2'
    kRnnType = 'lstm'
    kConvLstmFilterSize = 3
    kConvLstmHiddenDepth = 12
    kRnnHiddenDim = 512

    # Shared options
    kWeightDecay = 5e-5
    kBaseLearnRate = 1e-3
    kLearnRateDecay = 0.96
    kStepsPerLearnRateDecay = 5000
    kLossMixRatio = 1.0
    kMlpDropout = 0.5

    kNumMlpLayers = 2
    kMlpDepth = 6
    kMlpDropout = 0.5
    kDcnnFilterSize = '3,3,3,3,3,3'
    kDcnnDepth = '8,6,4,4,2,1'
    kDcnnPool = '2,2,2,2,2,1'
    kScoreMaxpool = 1

    # Attention-based model options
    kFilterWidth = 48
    kFilterHeight = 48
    kAttnBoxPaddingRatio = 0.2

    kCtrlCnnFilterSize = '3,3,3,3,3'
    kCtrlCnnDepth = '4,8,16,16,32'
    kCtrlCnnPool = '2,2,2,2,2'
    kAttnCnnFilterSize = '3,3,3'
    kAttnCnnDepth = '4,8,16'
    kAttnCnnPool = '2,2,2'
    kAttnDcnnFilterSize = '3,3,3,3'
    kAttnDcnnDepth = '16,8,4,1'
    kAttnDcnnPool = '2,2,2,1'

    kCtrlMlpDim = 256
    kNumCtrlMlpLayers = 2
    kCtrlRnnHiddenDim = 256
    kAttnRnnHiddenDim = 256
    kNumAttnMlpLayers = 2
    kAttnMlpDepth = 6
    kGtSelector = 'argmax'
    kKnobDecay = 0.9
    kStepsPerKnobDecay = 300
    kKnobBase = 1.0
    kKnobBoxOffset = 300
    kKnobSegmOffset = 500
    kGtBoxCtrNoise = 0.05
    kGtBoxPadNoise = 0.1
    kGtSegmNoise = 0.3
    kClipGradient = 1.0

    # Double attention model options
    kNumCtrlRNNIter = 5
    kNumGlimpseMlpLayers = 2

    # Model type
    parser.add_argument('-model', default='attention',
                        help='Which model to train')

    # Original model options
    parser.add_argument('-cnn_filter_size', default=kCnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-cnn_depth', default=kCnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-cnn_pool', default=kCnnPool,
                        help='Comma delimited integers')
    parser.add_argument('-dcnn_filter_size', default=kDcnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-dcnn_depth', default=kDcnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-dcnn_pool', default=kDcnnPool,
                        help='Comma delimited integers')
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
    parser.add_argument('-use_deconv', action='store_true',
                        help='Whether to use deconvolution layer to upsample.')
    parser.add_argument('-segm_dense_conn', action='store_true',
                        help='Whether to use dense connection to segment.')
    parser.add_argument('-add_skip_conn', action='store_true',
                        help='Whether to add skip connection in the DCNN.')
    parser.add_argument('-score_use_core', action='store_true',
                        help='Use core MLP network to predict score.')

    # Shared options
    parser.add_argument('-weight_decay', default=kWeightDecay, type=float,
                        help='Weight L2 regularization')
    parser.add_argument('-base_learn_rate', default=kBaseLearnRate,
                        type=float, help='Model learning rate')
    parser.add_argument('-learn_rate_decay', default=kLearnRateDecay,
                        type=float, help='Model learning rate decay')
    parser.add_argument('-steps_per_learn_rate_decay',
                        default=kStepsPerLearnRateDecay, type=int,
                        help='Steps every learning rate decay')
    parser.add_argument('-loss_mix_ratio', default=kLossMixRatio, type=float,
                        help='Mix ratio between segmentation and score loss')
    parser.add_argument('-segm_loss_fn', default='iou',
                        help=('Segmentation loss function, "iou", "wt_iou", '
                              '"wt_cov", or "bce"'))
    parser.add_argument('-mlp_dropout', default=kMlpDropout,
                        type=float, help='MLP dropout')
    parser.add_argument('-use_bn', action='store_true',
                        help='Whether to use batch normalization.')
    parser.add_argument('-no_cum_min', action='store_true',
                        help='Whether cumulative minimum. Default yes.')
    parser.add_argument('-fixed_order', action='store_true',
                        help='Fix the groundtruth order.')

    # Attention-based model options
    parser.add_argument('-filter_height', default=kFilterHeight, type=int,
                        help='Attention filter width')
    parser.add_argument('-filter_width', default=kFilterWidth, type=int,
                        help='Attention filter size')
    parser.add_argument('-ctrl_cnn_filter_size', default=kCtrlCnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-ctrl_cnn_depth', default=kCtrlCnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-ctrl_cnn_pool', default=kCtrlCnnPool,
                        help='Comma delimited integers')
    parser.add_argument('-attn_cnn_filter_size', default=kAttnCnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-attn_cnn_depth', default=kAttnCnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-attn_cnn_pool', default=kAttnCnnPool,
                        help='Comma delimited integers')
    parser.add_argument('-attn_dcnn_filter_size', default=kAttnDcnnFilterSize,
                        help='Comma delimited integers')
    parser.add_argument('-attn_dcnn_depth', default=kAttnDcnnDepth,
                        help='Comma delimited integers')
    parser.add_argument('-attn_dcnn_pool', default=kAttnDcnnPool,
                        help='Comma delimited integers')
    parser.add_argument('-ctrl_rnn_hid_dim', default=kCtrlRnnHiddenDim,
                        type=int, help='RNN hidden dimension')
    parser.add_argument('-attn_rnn_hid_dim', default=kAttnRnnHiddenDim,
                        type=int, help='RNN hidden dimension')
    parser.add_argument('-num_ctrl_mlp_layers', default=kNumCtrlMlpLayers,
                        type=int, help='Number of controller MLP layers')
    parser.add_argument('-ctrl_mlp_dim', default=kCtrlMlpDim,
                        type=int, help='Controller MLP dimension')
    parser.add_argument('-num_attn_mlp_layers', default=kNumAttnMlpLayers,
                        type=int, help='Number of attention MLP layers')
    parser.add_argument('-attn_mlp_depth', default=kAttnMlpDepth,
                        type=int, help='Attention MLP depth')
    parser.add_argument('-box_loss_fn', default='iou',
                        help='Box loss function, "iou" or "bce"')
    parser.add_argument('-use_gt_attn', action='store_true',
                        help='Whether to use ground truth attention.')
    parser.add_argument('-attn_box_padding_ratio',
                        default=kAttnBoxPaddingRatio, type=float,
                        help='Padding ratio of attention box')
    parser.add_argument('-use_attn_rnn', action='store_true',
                        help='Whether to use an inner RNN.')
    parser.add_argument('-use_canvas', action='store_true',
                        help='Whether to use a canvas to store.')
    parser.add_argument('-use_knob', action='store_true',
                        help='Whether to use a knob.')
    parser.add_argument('-knob_decay', default=kKnobDecay, type=float,
                        help='Knob decay factor.')
    parser.add_argument('-steps_per_knob_decay', default=kStepsPerKnobDecay,
                        type=int, help='Number of steps to decay knob.')
    parser.add_argument('-knob_base', default=kKnobBase, type=float,
                        help='Knob start rate.')
    parser.add_argument('-knob_box_offset', default=kKnobBoxOffset, type=int,
                        help='Number of steps when it starts to decay.')
    parser.add_argument('-knob_segm_offset', default=kKnobSegmOffset, type=int,
                        help='Number of steps when it starts to decay.')
    parser.add_argument('-knob_use_timescale', action='store_true',
                        help='Use time scale curriculum.')
    parser.add_argument('-gt_selector', default=kGtSelector,
                        help='greedy_match or argmax')
    parser.add_argument('-gt_box_ctr_noise', default=kGtBoxCtrNoise,
                        type=float, help='Groundtruth box center noise')
    parser.add_argument('-gt_box_pad_noise', default=kGtBoxPadNoise,
                        type=float, help='Groundtruth box padding noise')
    parser.add_argument('-gt_segm_noise', default=kGtSegmNoise,
                        type=float, help='Groundtruth segmentation noise')
    parser.add_argument('-downsample_canvas', action='store_true',
                        help='Whether downsample canvas to feed to Ctrl RNN')
    parser.add_argument('-pretrain_cnn', default=None,
                        help='Use pre-trained segmentation CNN')
    parser.add_argument('-cnn_share_weights', action='store_true',
                        help='Whether to share weights between CCNN and ACNN')
    parser.add_argument('-use_iou_box', action='store_true',
                        help='Whether to use box to calculate IoU')
    parser.add_argument('-clip_gradient', default=kClipGradient, type=float,
                        help='Largest gradient norm size')
    parser.add_argument('-squash_ctrl_params', action='store_true',
                        help='Whether to squash control parameters.')
    parser.add_argument('-fixed_gamma', action='store_true',
                        help='Fix the value of gamma.')

    # Double attention arguments
    parser.add_argument('-num_ctrl_rnn_iter', default=kNumCtrlRNNIter,
                        type=int, help='Number of control RNN iterations')
    parser.add_argument('-num_glimpse_mlp_layers', default=kNumGlimpseMlpLayers,
                        type=int, help='Number of glimpse MLP layers')
    pass


def _add_training_args(parser):
    # Default training options
    kNumSteps = 500000
    kStepsPerCkpt = 1000
    kStepsPerValid = 250
    kStepsPerTrainval = 100
    kStepsPerPlot = 50
    kStepsPerLog = 20
    kBatchSize = 32

    # Training options
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-steps_per_valid', default=kStepsPerValid,
                        type=int, help='Number of steps per validation')
    parser.add_argument('-steps_per_trainval', default=kStepsPerTrainval,
                        type=int, help='Number of steps per train validation')
    parser.add_argument('-steps_per_plot', default=kStepsPerPlot,
                        type=int, help='Number of steps per plot samples')
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
    parser.add_argument('-debug_bn', action='store_true',
                        help='Write out logs on batch normalization')
    parser.add_argument('-debug_act', action='store_true',
                        help='Write out logs on conv layer activation')
    parser.add_argument('-no_valid', action='store_true',
                        help='Use the whole training set.')
    parser.add_argument('-debug_weights', action='store_true',
                        help='Plot the weights')
    parser.add_argument('-debug_nan', action='store_true', help='Debug NaN')

    pass


def _make_model_opt(args):
    """Convert command-line arguments into model opt dict."""
    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1
        inp_height = kSynthShapeInpHeight
        inp_width = kSynthShapeInpWidth
        rnd_hflip = True
        rnd_vflip = True
        rnd_transpose = True
        rnd_colour = False
    elif args.dataset == 'cvppp':
        timespan = kCvpppNumObj + 1
        inp_height = kCvpppInpWidth
        inp_width = kCvpppInpHeight
        max_items_per_row = 8
        rnd_hflip = True
        rnd_vflip = True
        rnd_transpose = True
        rnd_colour = False
    elif args.dataset == 'kitti':
        timespan = kKittiNumObj + 1
        inp_height = kKittiInpHeight
        inp_width = kKittiInpWidth
        rnd_hflip = True
        rnd_vflip = False
        rnd_transpose = False
        rnd_colour = False
    else:
        raise Exception('Unknown dataset name')

    if args.model == 'attention' or args.model == 'double_attention':
        ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
        ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
        ccnn_depth_list = args.ctrl_cnn_depth.split(',')
        ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
        ccnn_pool_list = args.ctrl_cnn_pool.split(',')
        ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

        acnn_fsize_list = args.attn_cnn_filter_size.split(',')
        acnn_fsize_list = [int(fsize) for fsize in acnn_fsize_list]
        acnn_depth_list = args.attn_cnn_depth.split(',')
        acnn_depth_list = [int(depth) for depth in acnn_depth_list]
        acnn_pool_list = args.attn_cnn_pool.split(',')
        acnn_pool_list = [int(pool) for pool in acnn_pool_list]

        attn_dcnn_fsize_list = args.attn_dcnn_filter_size.split(',')
        attn_dcnn_fsize_list = [int(fsize) for fsize in attn_dcnn_fsize_list]
        attn_dcnn_depth_list = args.attn_dcnn_depth.split(',')
        attn_dcnn_depth_list = [int(depth) for depth in attn_dcnn_depth_list]
        attn_dcnn_pool_list = args.attn_dcnn_pool.split(',')
        attn_dcnn_pool_list = [int(pool) for pool in attn_dcnn_pool_list]

        model_opt = {
            'type': args.model,
            'inp_height': inp_height,
            'inp_width': inp_width,
            'inp_depth': 3,
            'padding': args.padding,
            'filter_height': args.filter_height,
            'filter_width': args.filter_width,
            'timespan': timespan,

            'ctrl_cnn_filter_size': ccnn_fsize_list,
            'ctrl_cnn_depth': ccnn_depth_list,
            'ctrl_cnn_pool': ccnn_pool_list,

            'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,
            'attn_rnn_hid_dim': args.attn_rnn_hid_dim,

            'attn_cnn_filter_size': acnn_fsize_list,
            'attn_cnn_depth': acnn_depth_list,
            'attn_cnn_pool': acnn_pool_list,

            'attn_dcnn_filter_size': attn_dcnn_fsize_list,
            'attn_dcnn_depth': attn_dcnn_depth_list,
            'attn_dcnn_pool': attn_dcnn_pool_list,

            'num_ctrl_mlp_layers': args.num_ctrl_mlp_layers,
            'ctrl_mlp_dim': args.ctrl_mlp_dim,

            'attn_mlp_depth': args.attn_mlp_depth,
            'num_attn_mlp_layers': args.num_attn_mlp_layers,
            'mlp_dropout': args.mlp_dropout,

            'weight_decay': args.weight_decay,
            'base_learn_rate': args.base_learn_rate,
            'learn_rate_decay': args.learn_rate_decay,
            'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
            'loss_mix_ratio': args.loss_mix_ratio,

            # Test arguments
            'segm_loss_fn': args.segm_loss_fn,
            'box_loss_fn': args.box_loss_fn,
            'use_bn': args.use_bn,
            'use_gt_attn': args.use_gt_attn,                      # DEPRECATED
            'attn_box_padding_ratio': args.attn_box_padding_ratio,
            'use_attn_rnn': args.use_attn_rnn,
            'use_canvas': args.use_canvas,
            'use_knob': args.use_knob,
            'knob_decay': args.knob_decay,
            'knob_base': args.knob_base,
            'steps_per_knob_decay': args.steps_per_knob_decay,
            'knob_box_offset': args.knob_box_offset,
            'knob_segm_offset': args.knob_segm_offset,
            'knob_use_timescale': args.knob_use_timescale,
            'gt_selector': args.gt_selector,
            'gt_box_ctr_noise': args.gt_box_ctr_noise,
            'gt_box_pad_noise': args.gt_box_pad_noise,
            'gt_segm_noise': args.gt_segm_noise,
            'downsample_canvas': args.downsample_canvas,
            'pretrain_cnn': args.pretrain_cnn,
            'cnn_share_weights': args.cnn_share_weights,
            'squash_ctrl_params': args.squash_ctrl_params,
            'use_iou_box': args.use_iou_box,
            'clip_gradient': args.clip_gradient,
            'fixed_order': args.fixed_order,
            'fixed_gamma': args.fixed_gamma,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour,
        }
        if args.model == 'double_attention':
            model_opt['num_ctrl_rnn_iter'] = args.num_ctrl_rnn_iter
            model_opt['num_glimpse_mlp_layers'] = args.num_glimpse_mlp_layers

    elif args.model == 'vanilla':
        cnn_fsize_list = args.cnn_filter_size.split(',')
        cnn_fsize_list = [int(fsize) for fsize in cnn_fsize_list]
        cnn_depth_list = args.cnn_depth.split(',')
        cnn_depth_list = [int(depth) for depth in cnn_depth_list]
        cnn_pool_list = args.cnn_pool.split(',')
        cnn_pool_list = [int(pool) for pool in cnn_pool_list]

        dcnn_fsize_list = args.dcnn_filter_size.split(',')
        dcnn_fsize_list = [int(fsize) for fsize in dcnn_fsize_list]
        dcnn_depth_list = args.dcnn_depth.split(',')
        dcnn_depth_list = [int(depth) for depth in dcnn_depth_list]
        dcnn_pool_list = args.dcnn_pool.split(',')
        dcnn_pool_list = [int(pool) for pool in dcnn_pool_list]

        model_opt = {
            'type': args.model,
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
            'cnn_filter_size': cnn_fsize_list,
            'cnn_depth': cnn_depth_list,
            'dcnn_filter_size': dcnn_fsize_list,
            'dcnn_depth': dcnn_depth_list,
            'rnn_type': args.rnn_type,
            'conv_lstm_filter_size': args.conv_lstm_filter_size,
            'conv_lstm_hid_depth': args.conv_lstm_hid_depth,
            'rnn_hid_dim': args.rnn_hid_dim,
            'mlp_depth': args.mlp_depth,
            'score_maxpool': args.score_maxpool,
            'num_mlp_layers': args.num_mlp_layers,
            'mlp_dropout': args.mlp_dropout,

            # Test arguments
            # 'cum_min': not args.no_cum_min,
            'cum_min': True,
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
            'score_use_core': True,
            'clip_gradient': args.clip_gradient,
            'fixed_order': args.fixed_order,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour
        }

    return model_opt


def _make_data_opt(args):
    """Make command-line arguments into data opt dict."""
    if args.dataset == 'synth_shape':
        timespan = args.max_num_objects + 1
        inp_height = kSynthShapeInpHeight
        inp_width = kSynthShapeInpWidth
    elif args.dataset == 'cvppp':
        timespan = kCvpppNumObj + 1
        inp_height = kCvpppInpHeight
        inp_width = kCvpppInpWidth
    elif args.dataset == 'kitti':
        timespan = kKittiNumObj + 1
        inp_height = kKittiInpHeight
        inp_width = kKittiInpWidth
    else:
        raise Exception('Unknown dataset name')

    if args.dataset == 'synth_shape':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'radius_upper': args.radius_upper,
            'radius_lower': args.radius_lower,
            'border_thickness': args.border_thickness,
            'max_num_objects': args.max_num_objects,
            'num_object_types': args.num_object_types,
            'center_var': args.center_var,
            'size_var': args.size_var,
            'num_train': args.num_ex,
            'num_valid': int(args.num_ex / 10),
            'has_valid': True
        }
    elif args.dataset == 'cvppp':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'num_train': None,
            'num_valid': None,
            'has_valid': not args.no_valid
        }
    elif args.dataset == 'kitti':
        data_opt = {
            'height': inp_height,
            'width': inp_width,
            'timespan': timespan,
            'num_train': args.num_ex,
            'num_valid': args.num_ex,
            'has_valid': True
        }

    return data_opt


def _make_train_opt(args):
    """Train opt"""
    train_opt = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt,
        'steps_per_valid': args.steps_per_valid,
        'steps_per_trainval': args.steps_per_trainval,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log,
        'debug_bn': args.debug_bn,
        'debug_act': args.debug_act,
        'has_valid': not args.no_valid,
        'results': args.results,
        'restore': args.restore,
        'save_ckpt': args.save_ckpt,
        'logs': args.logs,
        'gpu': args.gpu,
        'localhost': args.localhost,
        'debug_weights': args.debug_weights,
        'debug_nan': args.debug_nan
    }

    return train_opt


def _parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation + Attention')

    _add_dataset_args(parser)
    _add_model_args(parser)
    _add_training_args(parser)

    args = parser.parse_args()

    return args


def _get_model_id(task_name):
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)

    return model_id


def _get_ts_loggers(model_opt, restore_step=0, debug_bn=False, debug_weights=False):
    loggers = {}
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['segm_loss'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'segm_loss.csv'), ['train', 'valid'],
        name='Segmentation Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['iou'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'iou.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='IoU',
        buffer_size=1,
        restore_step=restore_step)
    loggers['wt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'wt_cov.csv'), ['train', 'valid'],
        name='Weighted Coverage',
        buffer_size=1,
        restore_step=restore_step)
    loggers['unwt_cov'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'unwt_cov.csv'), ['train', 'valid'],
        name='Unweighted Coverage',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dice'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dice.csv'), ['train', 'valid'],
        name='Dice',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dic'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dic.csv'), ['train', 'valid'],
        name='DiC',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dic_abs'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'dic_abs.csv'), ['train', 'valid'],
        name='|DiC|',
        buffer_size=1,
        restore_step=restore_step)
    loggers['learn_rate'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'learn_rate.csv'), 'learning rate',
        name='Learning rate',
        buffer_size=1,
        restore_step=restore_step)
    loggers['count_acc'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'count_acc.csv'), ['train', 'valid'],
        name='Count acc',
        buffer_size=1,
        restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
        name='Step time',
        buffer_size=1,
        restore_step=restore_step)

    if model_opt['type'] == 'attention' or \
            model_opt['type'] == 'double_attention':
        loggers['box_loss'] = TimeSeriesLogger(
            os.path.join(logs_folder, 'box_loss.csv'), ['train', 'valid'],
            name='Box Loss',
            buffer_size=1,
            restore_step=restore_step)
        loggers['gt_knob'] = TimeSeriesLogger(
            os.path.join(logs_folder, 'gt_knob.csv'),
            ['box', 'segmentation'],
            name='GT mix',
            buffer_size=1,
            restore_step=restore_step)
        loggers['attn_params'] = TimeSeriesLogger(
            os.path.join(logs_folder, 'attn_params.csv'),
            ['attn log gamma', 'box log gamma', 'out log gamma'],
            name='Attn params',
            buffer_size=1,
            restore_step=restore_step)

    if debug_weights:
        for layer_name, nlayers in zip(['glimpse_mlp', 'ctrl_mlp'],
                                       [model_opt['num_glimpse_mlp_layers'],
                                        model_opt['num_ctrl_mlp_layers']]):
            labels = []
            for ii in xrange(nlayers):
                labels.append('w_{}'.format(ii))
                labels.append('b_{}'.format(ii))
            loggers[layer_name] = TimeSeriesLogger(
                os.path.join(logs_folder,
                             '{}.csv'.format(layer_name)),
                labels,
                name='{} weights stats'.format(layer_name),
                buffer_size=1)
        for layer_name in ['ctrl_lstm']:
            labels = ['w_x', 'w_h', 'b']
            loggers[layer_name] = TimeSeriesLogger(
                os.path.join(logs_folder,
                             '{}.csv'.format(layer_name)),
                labels,
                name='{} weights stats'.format(layer_name),
                buffer_size=1)

    if debug_bn:
        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_attn_dcnn = len(model_opt['attn_dcnn_filter_size'])
        for sname, fname, num_layers in zip(
                ['ctrl_cnn', 'attn_cnn', 'attn_dcnn'],
                ['Ctrl CNN', 'Attn CNN', 'D-CNN'],
                [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
            for ii in xrange(num_layers):
                num_iter = model_opt['timespan']
                if 'num_ctrl_rnn_iter' in model_opt and \
                        sname == 'ctrl_cnn':
                    num_iter = model_opt['num_ctrl_rnn_iter']
                for tt in xrange(num_iter):
                    _bn_logger = TimeSeriesLogger(
                        os.path.join(
                            logs_folder,
                            '{}_{}_bn_{}.csv'.format(sname, ii, tt)),
                        ['train batch mean', 'valid batch mean',
                         'train batch var', 'valid batch var', 'ema mean',
                         'ema var'],
                        name='{} {} time {} batch norm stats'.format(
                            fname, ii, tt),
                        buffer_size=1)
                    loggers['bn_{}_{}_{}'.format(sname, ii, tt)] = \
                        _bn_logger

    return loggers


def _get_plot_loggers(model_opt, train_opt):
    samples = {}
    _ssets = ['train']
    if train_opt['has_valid']:
        _ssets.append('valid')
    for _set in _ssets:
        labels = ['input', 'output', 'total']
        if model_opt['type'] == 'attention' or \
                model_opt['type'] == 'double_attention':
            num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
            num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
            num_attn_dcnn = len(model_opt['attn_dcnn_filter_size'])
            labels.extend(['box', 'patch'])
        if args.debug_act:
            for _layer, _num in zip(
                    ['ccnn', 'acnn', 'adcnn'],
                    [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
                for ii in xrange(_num):
                    labels.append('{}_{}'.format(_layer, ii))
        for name in labels:
            key = '{}_{}'.format(name, _set)
            samples[key] = LazyRegisterer(
                os.path.join(logs_folder, '{}.png'.format(key)),
                'image', 'Samples {} {}'.format(name, _set))
    return samples


def _register_raw_logs(log_manager, log, model_opt, saver):
    log_manager.register(log.filename, 'plain', 'Raw logs')
    cmd_fname = os.path.join(logs_folder, 'cmd.log')
    with open(cmd_fname, 'w') as f:
        f.write(' '.join(sys.argv))
    log_manager.register(cmd_fname, 'plain', 'Command-line arguments')
    model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
    saver.save_opt(model_opt_fname, model_opt)
    log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

    pass


def _get_max_items_per_row(inp_height, inp_width):
    if inp_height == inp_width:
        return 8
    else:
        return 5


def _get_num_batch_valid(dataset_name):
    if dataset_name == 'synth_shape':
        return 5
    elif dataset_name == 'cvppp':
        return 5
    elif dataset_name == 'kitti':
        return 10
    else:
        raise Exception('Unknown dataset name')


if __name__ == '__main__':
    # Command-line arguments
    args = _parse_args()
    tf.set_random_seed(1234)
    saver = None
    train_opt = _make_train_opt(args)
    model_opt = _make_model_opt(args)
    data_opt = _make_data_opt(args)

    # Restore previously saved checkpoints.
    if train_opt['restore']:
        saver = Saver(train_opt['restore'])
        ckpt_info = saver.get_ckpt_info()
        model_opt = ckpt_info['model_opt']
        data_opt = ckpt_info['data_opt']
        ckpt_fname = ckpt_info['ckpt_fname']
        step = ckpt_info['step']
        model_id = ckpt_info['model_id']
        exp_folder = train_opt['restore']
    else:
        model_id = _get_model_id('rec_ins_segm')
        step = 0
        exp_folder = os.path.join(train_opt['results'], model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if model_opt['type'] == 'attention' or model_opt['type'] == 'double_attention':
        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_attn_dcnn = len(model_opt['attn_dcnn_filter_size'])

    if not train_opt['save_ckpt']:
        log.warning(
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

    # Logger
    if train_opt['logs']:
        logs_folder = train_opt['logs']
        logs_folder = os.path.join(logs_folder, model_id)
        log = logger.get(os.path.join(logs_folder, 'raw.log'))
    else:
        log = logger.get()

    # Log arguments
    log.log_args()

    # Set device
    if train_opt['gpu'] >= 0:
        device = '/gpu:{}'.format(train_opt['gpu'])
    else:
        device = '/cpu:0'

    # Train loop options
    log.info('Building model')
    m = get_model(model_opt, device=device)

    log.info('Loading dataset')
    dataset = get_dataset(args.dataset, data_opt)

    if model_opt['fixed_order']:
        dataset['train']['label_segmentation'] = base.sort_by_segm_size(
            dataset['train']['label_segmentation'])
        dataset['valid']['label_segmentation'] = base.sort_by_segm_size(
            dataset['valid']['label_segmentation'])

    sess = tf.Session()

    if args.restore:
        saver.restore(sess, ckpt_fname)
    else:
        sess.run(tf.initialize_all_variables())

    # Create time series loggers
    loggers = {}
    if train_opt['logs']:
        log_manager = LogManager(logs_folder)
        loggers = _get_ts_loggers(model_opt, debug_bn=train_opt['debug_bn'],
                                  debug_weights=train_opt['debug_weights'])
        _register_raw_logs(log_manager, log, model_opt, saver)
        samples = _get_plot_loggers(model_opt, train_opt)
        _log_url = 'http://{}/deep-dashboard?id={}'.format(
            train_opt['localhost'], model_id)
        log.info('Visualization can be viewed at: {}'.format(_log_url))

    batch_size = args.batch_size
    log.info('Batch size: {}'.format(batch_size))
    num_ex_train = dataset['train']['input'].shape[0]
    get_batch_train = _get_batch_fn(dataset['train'])
    log.info('Number of training examples: {}'.format(num_ex_train))

    if train_opt['has_valid']:
        num_ex_valid = dataset['valid']['input'].shape[0]
        get_batch_valid = _get_batch_fn(dataset['valid'])
        log.info('Number of validation examples: {}'.format(num_ex_valid))

    def run_samples():
        """Samples"""
        attn = model_opt['type'] == 'attention' or model_opt[
            'type'] == 'double_attention'

        def _run_samples(x, y, s, phase_train, fname_input, fname_output,
                         fname_total=None, fname_box=None, fname_patch=None,
                         fname_ccnn=None, fname_acnn=None, fname_attn_dcnn=None):

            _outputs = ['x_trans', 'y_gt_trans', 'y_out',
                        's_out', 'match']

            if attn:
                _outputs.extend(['attn_top_left', 'attn_bot_right',
                                 'attn_box', 'attn_box_gt', 'match_box'])

            _max_items = _get_max_items_per_row(x.shape[1], x.shape[2])

            if fname_patch:
                _outputs.append('x_patch')
            if fname_ccnn:
                [_outputs.append('h_ccnn_{}'.format(ii))
                 for ii in xrange(num_ctrl_cnn)]
                h_ccnn = [None] * num_ctrl_cnn
            if fname_acnn:
                [_outputs.append('h_acnn_{}'.format(ii))
                 for ii in xrange(num_attn_cnn)]
                h_acnn = [None] * num_attn_cnn
            if fname_attn_dcnn:
                [_outputs.append('h_attn_dcnn_{}'.format(ii))
                 for ii in xrange(num_attn_dcnn)]
                h_attn_dcnn = [None] * num_attn_dcnn

            _feed_dict = {m['x']: x, m['phase_train']: phase_train,
                          m['y_gt']: y, m['s_gt']: s}
            _r = _run_model(m, _outputs, _feed_dict)

            plot_input(fname_input, x=_r['x_trans'], y_gt=_r['y_gt_trans'],
                       s_gt=s, max_items_per_row=_max_items)

            if attn:
                plot_output(fname_output, y_out=_r['y_out'], s_out=_r['s_out'],
                            match=_r['match'],
                            attn=(_r['attn_top_left'], _r['attn_bot_right']),
                            max_items_per_row=_max_items)
            else:
                plot_output(fname_output, y_out=_r['y_out'], s_out=_r['s_out'],
                            match=_r['match'], max_items_per_row=_max_items)

            if fname_total:
                plot_total_instances(fname_total, y_out=_r['y_out'],
                                     s_out=_r['s_out'],
                                     max_items_per_row=_max_items)

            if fname_box:
                plot_output(fname_box, y_out=_r['attn_box'], s_out=_r['s_out'],
                            match=_r['match_box'],
                            attn=(_r['attn_top_left'], _r['attn_bot_right']),
                            max_items_per_row=_max_items)

            if fname_patch:
                pu.plot_thumbnails(fname_patch,
                                   _r['x_patch'][:, :, :, :, : 3],
                                   axis=1, max_items_per_row=8)

            if fname_ccnn:
                for ii in xrange(num_ctrl_cnn):
                    _h = _r['h_ccnn_{}'.format(ii)]
                    pu.plot_thumbnails(fname_ccnn[ii], _h[:, 0], axis=3,
                                       max_items_per_row=_max_items)

            if fname_acnn:
                for ii in xrange(num_attn_cnn):
                    _h = _r['h_acnn_{}'.format(ii)]
                    pu.plot_thumbnails(fname_acnn[ii], _h[:, 0], axis=3,
                                       max_items_per_row=8)

            if fname_attn_dcnn:
                for ii in xrange(num_attn_dcnn):
                    _h = _r['h_attn_dcnn_{}'.format(ii)]
                    pu.plot_thumbnails(fname_attn_dcnn[ii], _h[ii][:, 0],
                                       axis=3,
                                       max_items_per_row=8)

            pass

        # Plot some samples.
        _ssets = ['train']
        if train_opt['has_valid']:
            _ssets.append('valid')

        for _set in _ssets:
            _is_train = _set == 'train'
            _get_batch = get_batch_train if _is_train else get_batch_valid
            _num_ex = num_ex_train if _is_train else num_ex_valid
            log.info('Plotting {} samples'.format(_set))
            _x, _y, _s = _get_batch(
                np.arange(min(_num_ex, args.num_samples_plot)))

            if args.debug_act:
                fname_ccnn = [samples['ccnn_{}_{}'.format(
                    ii, _set)].get_fname() for ii in xrange(num_ctrl_cnn)]
                fname_acnn = [samples['acnn_{}_{}'.format(
                    ii, _set)].get_fname() for ii in xrange(num_attn_cnn)]
                fname_attn_dcnn = [samples['adcnn_{}_{}'.format(
                    ii, _set)].get_fname() for ii in xrange(num_attn_dcnn)]
            else:
                fname_ccnn = None
                fname_acnn = None
                fname_attn_dcnn = None

            if attn:
                fname_box = samples['box_{}'.format(_set)].get_fname()
                fname_patch = samples['patch_{}'.format(_set)].get_fname()
                names = ['input', 'output', 'total', 'box', 'patch']
            else:
                fname_box = None
                fname_patch = None
                names = ['input', 'output', 'total']

            _run_samples(
                _x, _y, _s, _is_train,
                fname_input=samples['input_{}'.format(_set)].get_fname(),
                fname_output=samples['output_{}'.format(_set)].get_fname(),
                fname_total=samples['total_{}'.format(_set)].get_fname(),
                fname_box=fname_box,
                fname_patch=fname_patch,
                fname_ccnn=fname_ccnn,
                fname_acnn=fname_acnn,
                fname_attn_dcnn=fname_attn_dcnn)

            if not samples['output_{}'.format(_set)].is_registered():
                for _name in names:
                    samples['{}_{}'.format(_name, _set)].register()

                if args.debug_act:
                    for _name, _num in zip(
                            ['ccnn', 'acnn', 'adcnn'],
                            [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
                        [samples[
                            '{}_{}_{}'.format(_name, ii, _set)].register()
                            for ii in xrange(_num)]
        pass

    def get_outputs_valid():
        _outputs = ['loss', 'conf_loss', 'segm_loss', 'iou_soft', 'iou_hard',
                    'count_acc', 'dice', 'dic', 'dic_abs', 'wt_cov_hard',
                    'unwt_cov_hard']

        if model_opt['type'] == 'attention' or model_opt['type'] == 'double_attention':
            _outputs.extend(['box_loss'])

        return _outputs

    def get_outputs_trainval():
        _outputs = ['loss', 'conf_loss', 'segm_loss', 'iou_soft', 'iou_hard',
                    'count_acc', 'dice', 'dic', 'dic_abs', 'wt_cov_hard',
                    'unwt_cov_hard', 'learn_rate']

        if train_opt['debug_weights']:
            for layer_name, nlayers in zip(['glimpse_mlp', 'ctrl_mlp'],
                                           [model_opt['num_glimpse_mlp_layers'],
                                            model_opt['num_ctrl_mlp_layers']]):
                for ii in xrange(nlayers):
                    _outputs.append('{}_w_{}_mean'.format(layer_name, ii))
                    _outputs.append('{}_b_{}_mean'.format(layer_name, ii))

            for layer_name in ['ctrl_lstm']:
                _outputs.append('{}_w_x_mean'.format(layer_name))
                _outputs.append('{}_w_h_mean'.format(layer_name))
                _outputs.append('{}_b_mean'.format(layer_name))

        if model_opt['type'] == 'attention' or model_opt['type'] == 'double_attention':
            _outputs.extend(['box_loss', 'gt_knob_prob_box',
                             'gt_knob_prob_segm', 'attn_lg_gamma_mean',
                             'attn_box_lg_gamma_mean', 'y_out_lg_gamma_mean'])
            # _outputs.extend(['box_loss', 'crnn_g_i_avg', 'crnn_g_f_avg',
            #                  'crnn_g_o_avg', 'gt_knob_prob_box',
            #                  'gt_knob_prob_segm', 'attn_lg_gamma_mean',
            #                  'attn_box_lg_gamma_mean', 'y_out_lg_gamma_mean'])

        return _outputs

    def get_outputs_bn():
        _outputs = []
        for _layer, _num in zip(
                ['ctrl_cnn', 'attn_cnn', 'attn_dcnn'],
                [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
            for ii in xrange(_num):
                for tt in xrange(model_opt['timespan']):
                    for _stat in ['bm', 'bv', 'em', 'ev']:
                        _outputs.append(
                            '{}_{}_{}_{}'.format(_layer, ii, _stat, tt))

        return _outputs

    def run_stats(step, num_batch, batch_iter, outputs, write_log, phase_train):
        """Validation"""
        nvalid = num_batch * batch_size
        r = {}

        for bb in xrange(num_batch):
            _x, _y, _s = batch_iter.next()
            _feed_dict = {m['x']: _x, m['phase_train']: phase_train,
                          m['y_gt']: _y, m['s_gt']: _s}
            _r = _run_model(m, outputs, _feed_dict)
            bat_sz = _x.shape[0]

            for key in _r.iterkeys():
                if key in r:
                    r[key] += _r[key] * bat_sz / nvalid
                else:
                    r[key] = _r[key] * bat_sz / nvalid

        log.info('{:d} loss {:.4f}'.format(step, r['loss']))
        write_log(step, loggers, r)

        pass

    def write_log_valid(step, loggers, r):
        attn = model_opt['type'] == 'attention' or model_opt[
            'type'] == 'double_attention'

        loggers['loss'].add(step, ['', r['loss']])
        loggers['conf_loss'].add(step, ['', r['conf_loss']])
        loggers['segm_loss'].add(step, ['', r['segm_loss']])
        if attn:
            loggers['box_loss'].add(step, ['', r['box_loss']])
        loggers['iou'].add(step, ['', r['iou_soft'], '', r['iou_hard']])
        loggers['wt_cov'].add(step, ['', r['wt_cov_hard']])
        loggers['unwt_cov'].add(step, ['', r['unwt_cov_hard']])
        loggers['dice'].add(step, ['', r['dice']])
        loggers['count_acc'].add(step, ['', r['count_acc']])
        loggers['dic'].add(step, ['', r['dic']])
        loggers['dic_abs'].add(step, ['', r['dic_abs']])

        # Batch normalization stats.
        if train_opt['debug_bn']:
            for _layer, _num in zip(
                    ['ctrl_cnn', 'attn_cnn', 'attn_dcnn'],
                    [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
                for ii in xrange(_num):
                    num_iter = model_opt['timespan']
                    if 'num_ctrl_rnn_iter' in model_opt and \
                            _layer == 'ctrl_cnn':
                        num_iter = model_opt['num_ctrl_rnn_iter']
                    for tt in xrange(num_iter):
                        _prefix = '{}_{}_{{}}_{}'.format(_layer, ii, tt)
                        _output = ['', r[_prefix.format('bm')],
                                   '', r[_prefix.format('bv')], '', '']
                        loggers['bn_{}_{}_{}'.format(
                            _layer, ii, tt)].add(step, _output)

        pass

    def write_log_trainval(step, loggers, r):
        attn = model_opt['type'] == 'attention' or model_opt[
            'type'] == 'double_attention'

        loggers['loss'].add(step, [r['loss'], ''])
        loggers['conf_loss'].add(step, [r['conf_loss'], ''])
        loggers['segm_loss'].add(step, [r['segm_loss'], ''])
        if attn:
            loggers['box_loss'].add(step, [r['box_loss'], ''])

        loggers['iou'].add(step, [r['iou_soft'], '', r['iou_hard'], ''])
        loggers['wt_cov'].add(step, [r['wt_cov_hard'], ''])
        loggers['unwt_cov'].add(step, [r['unwt_cov_hard'], ''])
        loggers['dice'].add(step, [r['dice'], ''])
        loggers['count_acc'].add(step, [r['count_acc'], ''])
        loggers['dic'].add(step, [r['dic'], ''])
        loggers['dic_abs'].add(step, [r['dic_abs'], ''])
        if attn:
            # loggers['crnn'].add(step, [r['crnn_g_i_avg'], r['crnn_g_f_avg'],
            #                            r['crnn_g_o_avg']])
            loggers['gt_knob'].add(step, [r['gt_knob_prob_box'],
                                          r['gt_knob_prob_segm']])
            loggers['attn_params'].add(step, [r['attn_lg_gamma_mean'],
                                              r['attn_box_lg_gamma_mean'],
                                              r['y_out_lg_gamma_mean']])
        loggers['learn_rate'].add(step, r['learn_rate'])

        if train_opt['debug_weights']:
            for layer_name, nlayers in zip(['glimpse_mlp', 'ctrl_mlp'],
                                           [model_opt['num_glimpse_mlp_layers'],
                                            model_opt['num_ctrl_mlp_layers']]):
                _output = []
                for ii in xrange(nlayers):
                    _output.append(r['{}_w_{}_mean'.format(layer_name, ii)])
                    _output.append(r['{}_b_{}_mean'.format(layer_name, ii)])
                loggers[layer_name].add(step, _output)

            for layer_name in ['ctrl_lstm']:
                _output = [r['{}_w_x_mean'.format(layer_name)],
                           r['{}_w_h_mean'.format(layer_name)],
                           r['{}_b_mean'.format(layer_name)]]
                loggers[layer_name].add(step, _output)

        # Batch normalization stats.
        if train_opt['debug_bn']:
            for _layer, _num in zip(
                    ['ctrl_cnn', 'attn_cnn', 'attn_dcnn'],
                    [num_ctrl_cnn, num_attn_cnn, num_attn_dcnn]):
                for ii in xrange(_num):
                    num_iter = model_opt['timespan']
                    if 'num_ctrl_rnn_iter' in model_opt and \
                            _layer == 'ctrl_cnn':
                        num_iter = model_opt['num_ctrl_rnn_iter']
                    for tt in xrange(num_iter):
                        _prefix = '{}_{}_{{}}_{}'.format(
                            _layer, ii, tt)
                        _output = [r[_prefix.format('bm')], '',
                                   r[_prefix.format('bv')], '',
                                   r[_prefix.format('em')],
                                   r[_prefix.format('ev')]]
                        loggers['bn_{}_{}_{}'.format(
                            _layer, ii, tt)].add(step, _output)

        pass

    def train_step(step, x, y, s, debug_nan=False):
        """Train step"""

        def check_nan(r):
            # Check NaN.
            if np.isnan(r['loss']):
                log.error('NaN occurred. Saving last step.')
                saver.save(sess, global_step=step)
                input_file = h5py.File(
                    os.path.join(exp_folder, 'nan_input.h5'))
                input_file['x'] = x
                input_file['y'] = y
                input_file['s'] = s
                raise Exception('NaN')

        if debug_nan:
            _start_time = time.time()
            _outputs = ['loss']
            _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                          m['s_gt']: s}
            r = _run_model(m, _outputs, _feed_dict)
            loss = r['loss']
            check_nan(r)

            _outputs = ['train_step']
            _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                          m['s_gt']: s}
            _start_time = time.time()
            r = _run_model(m, _outputs, _feed_dict)
            _step_time = (time.time() - _start_time) * 1000

            # Print statistics.
            if step % train_opt['steps_per_log'] == 0:
                log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, loss,
                                                              _step_time))
                loggers['loss'].add(step, [loss, ''])
                loggers['step_time'].add(step, _step_time)
        else:
            _start_time = time.time()
            _outputs = ['loss', 'train_step']
            _feed_dict = {m['x']: x, m['phase_train']: True, m['y_gt']: y,
                          m['s_gt']: s}
            r = _run_model(m, _outputs, _feed_dict)
            _step_time = (time.time() - _start_time) * 1000
            check_nan(r)

            # Print statistics.
            if step % train_opt['steps_per_log'] == 0:
                log.info('{:d} loss {:.4f} t {:.2f}ms'.format(step, r['loss'],
                                                              _step_time))
                loggers['loss'].add(step, [r['loss'], ''])
                loggers['step_time'].add(step, _step_time)

        pass

    def train_loop(step=0):
        """Train loop"""
        if train_opt['has_valid']:
            batch_iter_valid = BatchIterator(num_ex_valid,
                                             batch_size=batch_size,
                                             get_fn=get_batch_valid,
                                             cycle=True,
                                             progress_bar=False)
            outputs_valid = get_outputs_valid()
        num_batch_valid = _get_num_batch_valid(args.dataset)
        batch_iter_trainval = BatchIterator(num_ex_train,
                                            batch_size=batch_size,
                                            get_fn=get_batch_train,
                                            cycle=True,
                                            progress_bar=False)
        outputs_trainval = get_outputs_trainval()
        if train_opt['debug_bn']:
            if train_opt['has_valid']:
                outputs_valid.extend(get_outputs_bn())
            outputs_trainval.extend(get_outputs_bn())

        for _x, _y, _s in BatchIterator(num_ex_train,
                                        batch_size=batch_size,
                                        get_fn=get_batch_train,
                                        cycle=True,
                                        progress_bar=False):
            # Run validation stats
            if train_opt['has_valid']:
                if step % train_opt['steps_per_valid'] == 0:
                    log.info('Running validation')
                    run_stats(step, num_batch_valid, batch_iter_valid,
                              outputs_valid, write_log_valid, False)
                    pass

            # Train stats
            if step % train_opt['steps_per_trainval'] == 0:
                log.info('Running train validation')
                run_stats(step, num_batch_valid, batch_iter_trainval,
                          outputs_trainval, write_log_trainval, True)
                pass

            # Plot samples
            if step % train_opt['steps_per_plot'] == 0:
                run_samples()
                pass

            # Train step
            train_step(step, _x, _y, _s, debug_nan=train_opt['debug_nan'])

            # Model ID reminder
            if step % (10 * train_opt['steps_per_log']) == 0:
                log.info('model id {}'.format(model_id))
                pass

            # Save model
            if args.save_ckpt and step % train_opt['steps_per_ckpt'] == 0:
                saver.save(sess, global_step=step)
                pass

            step += 1

            # Termination
            if step > train_opt['num_steps']:
                break

        pass

    train_loop(step=step)

    sess.close()
    for logger in loggers.itervalues():
        logger.close()

    pass
