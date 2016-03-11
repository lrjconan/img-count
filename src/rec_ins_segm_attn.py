"""
Recurrent instance segmentation with attention.

Usage: python rec_ins_segm_attn.py --help
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
from data_api import kitti

from utils import log_manager
from utils import logger
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.saver import Saver
from utils.time_series_logger import TimeSeriesLogger
from utils import plot_utils

import rec_ins_segm_models as models

log = logger.get()


def plot_total_instances(fname, y_out, s_out, max_items_per_row=9):
    """Plot cumulative image with different colour at each timestep.

    Args:
        y_out: [B, T, H, W]
    """
    num_ex = y_out.shape[0]
    num_items = y_out.shape[1]
    num_row, num_col, calc = plot_utils.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    plot_utils.set_axis_off(axarr, num_row, num_col)

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


def plot_thumbnails(fname, img, axis, max_items_per_row=9):
    """Plot activation map.

    Args:
        img: [B, T, H, W, 3] or [B, H, W, D]
    """
    num_ex = img.shape[0]
    num_items = img.shape[axis]
    num_row, num_col, calc = plot_utils.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    plot_utils.set_axis_off(axarr, num_row, num_col)

    for ii in xrange(num_ex):
        for jj in xrange(num_items):
            row, col = calc(ii, jj)
            if axis == 3:
                x = img[ii, :, :, jj]
            elif axis == 1:
                x = img[ii, jj]
            if num_col > 1:
                ax = axarr[row, col]
            else:
                ax = axarr[row]
            ax.imshow(x)
            ax.text(0, -0.5, '[{:.2g}, {:.2g}]'.format(
                np.min(x), np.max(x)), color=(0, 0, 0), size=8)

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fname, dpi=150)
    plt.close('all')

    pass


def plot_input(fname, x, y_gt, s_gt, max_items_per_row=9):
    """Plot input, transformed input and output groundtruth sequence.
    """
    num_ex = y_gt.shape[0]
    num_items = y_gt.shape[1]
    num_row, num_col, calc = plot_utils.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    plot_utils.set_axis_off(axarr, num_row, num_col)
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
    num_row, num_col, calc = plot_utils.calc_row_col(
        num_ex, num_items, max_items_per_row=max_items_per_row)

    f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
    cmap = ['r', 'y', 'c', 'g', 'm']

    if attn:
        attn_top_left_y = attn[0][:, :, 0]
        attn_top_left_x = attn[0][:, :, 1]
        attn_bot_right_y = attn[1][:, :, 0]
        attn_bot_right_x = attn[1][:, :, 1]
        attn_ctr_y = attn[2][:, :, 0]
        attn_ctr_x = attn[2][:, :, 1]
        attn_delta_y = attn[3][:, :, 0]
        attn_delta_x = attn[3][:, :, 1]

    plot_utils.set_axis_off(axarr, num_row, num_col)

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


def get_dataset(dataset_name, opt, num_train=-1, num_valid=-1):
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
    elif dataset_name == 'kitti':
        if os.path.exists('/u/mren'):
            dataset_folder = '/ais/gobi3/u/mren/data/kitti'
        else:
            dataset_folder = '/home/mren/data/kitti'
        opt['timespan'] = 18
        opt['num_examples'] = num_train
        dataset['train'] = kitti.get_dataset(dataset_folder, opt, split='train')
        opt['num_examples'] = num_valid
        dataset['valid'] = kitti.get_dataset(dataset_folder, opt, split='valid')
    else:
        raise Exception('Unknown dataset "{}"'.format(dataset_name))

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
    kWeightDecay = 5e-5
    kBaseLearnRate = 1e-3
    kLearnRateDecay = 0.96
    kStepsPerLearnRateDecay = 5000
    kStepsPerLog = 10
    kLossMixRatio = 1.0
    kBoxLossCoeffDecay = 0.7
    kStepsPerBoxLossCoeffDecay = 2000
    kKnobDecay = 0.9
    kStepsPerKnobDecay = 300
    kKnobBase = 1.0
    kKnobBoxOffset = 300
    kKnobSegmOffset = 500

    kNumCtrlConv = 5
    kNumAttnConv = 3

    kAttnSize = 48

    kCtrlCnnFilterSize = [3, 3, 3, 3, 3]
    kCtrlCnnDepth = [4, 8, 8, 12, 16]
    kAttnCnnFilterSize = [3, 3, 3]
    kAttnCnnDepth = [4, 8, 16]

    kCtrlMlpDim = 256
    kNumCtrlMlpLayers = 1

    kCtrlRnnHiddenDim = 256
    kAttnRnnHiddenDim = 256

    kNumAttnMlpLayers = 2
    kAttnMlpDepth = 6

    kMlpDropout = 0.5
    kDcnnFilterSize = [3, 3, 3, 3]
    kDcnnDepth = [1, 4, 8, 16]

    kAttnBoxPaddingRatio = 0.2

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
    parser.add_argument('-num_ctrl_conv', default=kNumCtrlConv, type=int,
                        help='Number of controller convolutional layers')
    parser.add_argument('-num_attn_conv', default=kNumAttnConv, type=int,
                        help='Number of attention convolutional layers')
    parser.add_argument('-attn_size', default=kAttnSize, type=int,
                        help='Attention size')

    for ii in xrange(len(kCtrlCnnFilterSize)):
        parser.add_argument('-ctrl_cnn_{}_filter_size'.format(ii + 1),
                            default=kCtrlCnnFilterSize[ii], type=int,
                            help='Controller CNN layer {} filter size'.format(ii + 1))
        parser.add_argument('-ctrl_cnn_{}_depth'.format(ii + 1),
                            default=kCtrlCnnDepth[ii], type=int,
                            help='Controller CNN layer {} depth'.format(ii + 1))

    for ii in xrange(len(kAttnCnnFilterSize)):
        parser.add_argument('-attn_cnn_{}_filter_size'.format(ii + 1),
                            default=kAttnCnnFilterSize[ii], type=int,
                            help='Attention CNN layer {} filter size'.format(ii + 1))
        parser.add_argument('-attn_cnn_{}_depth'.format(ii + 1),
                            default=kAttnCnnDepth[ii], type=int,
                            help='Attention CNN layer {} depth'.format(ii + 1))

    for ii in xrange(len(kDcnnFilterSize)):
        parser.add_argument('-dcnn_{}_filter_size'.format(ii),
                            default=kDcnnFilterSize[ii], type=int,
                            help='DCNN layer {} filter size'.format(ii))
        parser.add_argument('-dcnn_{}_depth'.format(ii),
                            default=kDcnnDepth[ii], type=int,
                            help='DCNN layer {} depth'.format(ii))

    parser.add_argument('-ctrl_rnn_hid_dim', default=kCtrlRnnHiddenDim,
                        type=int, help='RNN hidden dimension')
    parser.add_argument('-attn_rnn_hid_dim', default=kCtrlRnnHiddenDim,
                        type=int, help='RNN hidden dimension')

    parser.add_argument('-num_ctrl_mlp_layers', default=kNumCtrlMlpLayers,
                        type=int, help='Number of controller MLP layers')
    parser.add_argument('-ctrl_mlp_dim', default=kCtrlMlpDim,
                        type=int, help='Controller MLP dimension')
    parser.add_argument('-num_attn_mlp_layers', default=kNumAttnMlpLayers,
                        type=int, help='Number of attention MLP layers')
    parser.add_argument('-attn_mlp_depth', default=kAttnMlpDepth,
                        type=int, help='Attntion MLP depth')

    parser.add_argument('-mlp_dropout', default=kMlpDropout,
                        type=float, help='MLP dropout')

    # Extra model options (beta)
    parser.add_argument('-segm_loss_fn', default='iou',
                        help='Segmentation loss function, "iou" or "bce"')
    parser.add_argument('-box_loss_fn', default='iou',
                        help='Box loss function, "iou" or "bce"')
    parser.add_argument('-use_bn', action='store_true',
                        help='Whether to use batch normalization.')
    parser.add_argument('-use_gt_attn', action='store_true',
                        help='Whether to use ground truth attention.')
    parser.add_argument('-attn_box_padding_ratio',
                        default=kAttnBoxPaddingRatio, type=float,
                        help='Padding ratio of attention box')
    parser.add_argument('-steps_per_box_loss_coeff_decay',
                        default=kStepsPerBoxLossCoeffDecay, type=int,
                        help='Number of steps to decay box loss coefficient.')
    parser.add_argument('-box_loss_coeff_decay', default=kBoxLossCoeffDecay,
                        type=float, help='Box loss coefficient decay factor.')
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

    # Training options
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-steps_per_valid', default=kStepsPerValid,
                        type=int, help='Number of steps per validation')
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

        ctrl_cnn_filter_size_all = [args.ctrl_cnn_1_filter_size,
                                    args.ctrl_cnn_2_filter_size,
                                    args.ctrl_cnn_3_filter_size,
                                    args.ctrl_cnn_4_filter_size,
                                    args.ctrl_cnn_5_filter_size]
        ctrl_cnn_depth_all = [args.ctrl_cnn_1_depth,
                              args.ctrl_cnn_2_depth,
                              args.ctrl_cnn_3_depth,
                              args.ctrl_cnn_4_depth,
                              args.ctrl_cnn_5_depth]

        attn_cnn_filter_size_all = [args.attn_cnn_1_filter_size,
                                    args.attn_cnn_2_filter_size,
                                    args.attn_cnn_3_filter_size]
        attn_cnn_depth_all = [args.attn_cnn_1_depth,
                              args.attn_cnn_2_depth,
                              args.attn_cnn_3_depth]

        dcnn_filter_size_all = [args.dcnn_0_filter_size,
                                args.dcnn_1_filter_size,
                                args.dcnn_2_filter_size,
                                args.dcnn_3_filter_size]
        dcnn_depth_all = [args.dcnn_0_depth,
                          args.dcnn_1_depth,
                          args.dcnn_2_depth,
                          args.dcnn_3_depth]

        if args.dataset == 'synth_shape':
            timespan = args.max_num_objects + 1
            inp_height = 224
            inp_width = 224
            max_items_per_row = 8
            rnd_hflip = True
            rnd_vflip = True
            rnd_transpose = True
            rnd_colour = False
        elif args.dataset == 'cvppp':
            timespan = 21
            inp_height = 224
            inp_width = 224
            max_items_per_row = 8
            rnd_hflip = True
            rnd_vflip = True
            rnd_transpose = True
            rnd_colour = False
        elif args.dataset == 'kitti':
            timespan = 18
            inp_height = 128
            inp_width = 448
            max_items_per_row = 5
            rnd_hflip = True
            rnd_vflip = False
            rnd_transpose = False
            rnd_colour = True
        else:
            raise Exception('Unknown dataset name')

        model_opt = {
            'inp_height': inp_height,
            'inp_width': inp_width,
            'inp_depth': 3,
            'padding': args.padding,
            'attn_size': args.attn_size,
            'timespan': timespan,

            'ctrl_cnn_filter_size': ctrl_cnn_filter_size_all[: args.num_ctrl_conv],
            'ctrl_cnn_depth': ctrl_cnn_depth_all[: args.num_ctrl_conv],
            'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,

            'attn_cnn_filter_size': attn_cnn_filter_size_all[: args.num_attn_conv],
            'attn_cnn_depth': attn_cnn_depth_all[: args.num_attn_conv],

            'attn_rnn_hid_dim': args.attn_rnn_hid_dim,

            'dcnn_filter_size': dcnn_filter_size_all[: args.num_attn_conv + 1][::-1],
            'dcnn_depth': dcnn_depth_all[: args.num_attn_conv + 1][::-1],

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
            'use_gt_attn': args.use_gt_attn,
            'attn_box_padding_ratio': args.attn_box_padding_ratio,
            'box_loss_coeff_decay': args.box_loss_coeff_decay,
            'steps_per_box_loss_coeff_decay': args.steps_per_box_loss_coeff_decay,
            'use_attn_rnn': args.use_attn_rnn,
            'use_canvas': args.use_canvas,
            'use_knob': args.use_knob,
            'knob_decay': args.knob_decay,
            'knob_base': args.knob_base,
            'steps_per_knob_decay': args.steps_per_knob_decay,
            'knob_box_offset': args.knob_box_offset,
            'knob_segm_offset': args.knob_segm_offset,
            'knob_use_timescale': args.knob_use_timescale,

            'rnd_hflip': rnd_hflip,
            'rnd_vflip': rnd_vflip,
            'rnd_transpose': rnd_transpose,
            'rnd_colour': rnd_colour
        }
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
            'size_var': args.size_var
        }
        step = 0
        exp_folder = os.path.join(args.results, model_id)
        saver = Saver(exp_folder, model_opt=model_opt, data_opt=data_opt)

    if not args.save_ckpt:
        log.warning(
            'Checkpoints saving is turned off. Use -save_ckpt flag to save.')

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
        'steps_per_valid': args.steps_per_valid,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log
    }

    log.info('Building model')
    m = models.get_model('attention', model_opt, device=device)

    log.info('Loading dataset')
    if args.dataset == 'synth_shape':
        dataset = get_dataset(args.dataset, data_opt,
                              args.num_ex, int(args.num_ex / 10))
    elif args.dataset == 'cvppp':
        dataset = get_dataset(args.dataset, data_opt)
    elif args.dataset == 'kitti':
        dataset = get_dataset(args.dataset, data_opt, args.num_ex, args.num_ex)

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
        conf_loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'conf_loss.csv'), ['train', 'valid'],
            name='Confidence Loss',
            buffer_size=1)
        segm_loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'segm_loss.csv'), ['train', 'valid'],
            name='Segmentation Loss',
            buffer_size=1)
        box_loss_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'box_loss.csv'), ['train', 'valid'],
            name='Box Loss',
            buffer_size=1)
        iou_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'iou.csv'),
            ['train soft', 'valid soft', 'train hard', 'valid hard'],
            name='IoU',
            buffer_size=1)
        dice_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'dice.csv'),
            ['train', 'valid'],
            name='Dice',
            buffer_size=1)
        dic_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'dic.csv'),
            ['train', 'valid'],
            name='DiC',
            buffer_size=1)
        dic_abs_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'dic_abs.csv'),
            ['train', 'valid'],
            name='|DiC|',
            buffer_size=1)
        learn_rate_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'learn_rate.csv'),
            'learning rate',
            name='Learning rate',
            buffer_size=1)
        count_acc_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'count_acc.csv'),
            ['train', 'valid'],
            name='Count acc',
            buffer_size=1)
        box_loss_coeff_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'box_loss_coeff.csv'),
            'box loss coeff',
            name='Box Loss Coeff',
            buffer_size=1)
        step_time_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'step_time.csv'), 'step time (ms)',
            name='Step time',
            buffer_size=1)
        crnn_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'crnn.csv'),
            ['input gate', 'forget gate', 'output gate'],
            name='Ctrl RNN',
            buffer_size=1)
        gt_knob_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'gt_knob.csv'),
            ['box', 'segmentation'],
            name='GT mix',
            buffer_size=1)
        attn_params_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'attn_params.csv'),
            ['attn log gamma', 'attn log var', 'box log gamma',
             'out log gamma', 'box beta', 'out beta'],
            name='Attn params',
            buffer_size=1)

        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])

        bn_loggers = {}
        for sname, fname, num_layers in zip(
                ['ccnn', 'acnn', 'dcnn'],
                ['Ctrl CNN', 'Attn CNN', 'D-CNN'],
                [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
            for ii in xrange(num_layers):
                for tt in xrange(model_opt['timespan']):
                    _bn_logger = TimeSeriesLogger(
                        os.path.join(logs_folder,
                                     '{}_{}_bn_{}.csv'.format(sname, ii, tt)),
                        ['train batch mean', 'valid batch mean',
                         'train batch var', 'valid batch var', 'ema mean',
                         'ema var'],
                        name='{} {} time {} batch norm stats'.format(
                            fname, ii, tt),
                        buffer_size=1)
                    bn_loggers['{}_{}_{}'.format(sname, ii, tt)] = _bn_logger

        acnn_weights_labels = []
        for ii in xrange(num_attn_cnn):
            acnn_weights_labels.append('|w| {}'.format(ii))
            acnn_weights_labels.append('|b| {}'.format(ii))
        acnn_weights_logger = TimeSeriesLogger(
            os.path.join(logs_folder, 'acnn_weights.csv'),
            acnn_weights_labels,
            name='Attn CNN weights mean',
            buffer_size=1)

        sparsity_loggers = {}
        for sname, fname, num_layers in zip(
                ['ccnn', 'acnn', 'dcnn'],
                ['Ctrl CNN', 'Attn CNN', 'D-CNN'],
                [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
            sparsity_labels = []
            for ii in xrange(num_layers):
                sparsity_labels.append('layer {}'.format(ii))
            sparsity_loggers[sname] = TimeSeriesLogger(
                os.path.join(logs_folder, '{}_sparsity.csv'.format(sname)),
                sparsity_labels,
                name='{} sparsity'.format(fname),
                buffer_size=1)

        log_manager.register(log.filename, 'plain', 'Raw logs')

        model_opt_fname = os.path.join(logs_folder, 'model_opt.yaml')
        saver.save_opt(model_opt_fname, model_opt)
        log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

        samples = {}
        for _set in ['train', 'valid']:
            labels = ['input', 'output', 'total', 'box', 'patch']
            for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                for ii in xrange(_num):
                    labels.append('{}_{}'.format(_layer, ii))
            for name in labels:
                key = '{}_{}'.format(name, _set)
                samples[key] = LazyRegisterer(
                    os.path.join(logs_folder, '{}.png'.format(key)),
                    'image', 'Samples {} {}'.format(name, _set))
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
        def _run_samples(x, y, s, phase_train, fname_input, fname_output,
                         fname_total=None, fname_box=None, fname_patch=None,
                         fname_ccnn=None, fname_acnn=None, fname_dcnn=None):
            results_list = [m['x_trans'], m['y_gt_trans'], m['y_out'],
                            m['s_out'], m['match'],
                            m['attn_top_left'], m['attn_bot_right'],
                            m['attn_ctr'], m['attn_delta'],
                            m['attn_box'], m['attn_box_gt'], m['match_box']]
            offset = len(results_list)
            if fname_patch:
                results_list.append(m['x_patch'])
            if fname_ccnn:
                results_list.extend(m['h_ccnn'])
            if fname_acnn:
                results_list.extend(m['h_acnn'])
            if fname_dcnn:
                results_list.extend(m['h_dcnn'])
            results = sess.run(results_list,
                               feed_dict={
                                   m['x']: x,
                                   m['phase_train']: phase_train,
                                   m['y_gt']: y,
                                   m['s_gt']: s
                               })
            x_trans = results[0]
            y_gt_trans = results[1]
            y_out = results[2]
            s_out = results[3]
            match = results[4]
            atl = results[5]
            abr = results[6]
            ac = results[7]
            ad = results[8]
            abox = results[9]
            abox_gt = results[10]
            match_box = results[11]
            h_ccnn = [None] * num_ctrl_cnn
            h_acnn = [None] * num_attn_cnn
            h_dcnn = [None] * num_dcnn

            if fname_patch:
                x_patch = results[offset]
                offset += 1
            if fname_ccnn:
                for ii in xrange(num_ctrl_cnn):
                    h_ccnn[ii] = results[offset]
                    offset += 1
            if fname_acnn:
                for ii in xrange(num_attn_cnn):
                    h_acnn[ii] = results[offset]
                    offset += 1
            if fname_dcnn:
                for ii in xrange(num_dcnn):
                    h_dcnn[ii] = results[offset]
                    offset += 1

            plot_input(fname_input, x=x_trans, y_gt=y_gt_trans, s_gt=s,
                       max_items_per_row=max_items_per_row)

            plot_output(fname_output, y_out=y_out, s_out=s_out, match=match,
                        attn=(atl, abr, ac, ad),
                        max_items_per_row=max_items_per_row)

            if fname_total:
                plot_total_instances(fname_total, y_out=y_out, s_out=s_out,
                                     max_items_per_row=max_items_per_row)

            if fname_box:
                plot_output(fname_box, y_out=abox, s_out=s_out,
                            match=match_box, attn=(atl, abr, ac, ad),
                            max_items_per_row=max_items_per_row)

            if fname_patch:
                plot_thumbnails(fname_patch, x_patch[:, :, :, :, : 3], axis=1,
                                max_items_per_row=8)

            if fname_ccnn:
                for ii in xrange(num_ctrl_cnn):
                    plot_thumbnails(fname_ccnn[ii], h_ccnn[ii][:, 0], axis=3,
                                    max_items_per_row=max_items_per_row)

            if fname_acnn:
                for ii in xrange(num_attn_cnn):
                    plot_thumbnails(fname_acnn[ii], h_acnn[ii][:, 0], axis=3,
                                    max_items_per_row=8)

            if fname_dcnn:
                for ii in xrange(num_dcnn):
                    plot_thumbnails(fname_dcnn[ii], h_dcnn[ii][:, 0], axis=3,
                                    max_items_per_row=8)

        if args.logs:
            # Plot some samples.
            for _set in ['valid', 'train']:
                _is_train = _set == 'train'
                _get_batch = get_batch_train if _is_train else get_batch_valid
                _num_ex = num_ex_train if _is_train else num_ex_valid
                log.info('Plotting {} samples'.format(_set))
                _x, _y, _s = _get_batch(
                    np.arange(min(_num_ex, args.num_samples_plot)))
                _run_samples(
                    _x, _y, _s, _is_train,
                    fname_input=samples['input_{}'.format(_set)].get_fname(),
                    fname_output=samples['output_{}'.format(_set)].get_fname(),
                    fname_total=samples['total_{}'.format(_set)].get_fname(),
                    fname_box=samples['box_{}'.format(_set)].get_fname(),
                    fname_patch=samples['patch_{}'.format(_set)].get_fname(),
                    fname_ccnn=[samples['ccnn_{}_{}'.format(
                        ii, _set)].get_fname() for ii in xrange(num_ctrl_cnn)],
                    fname_acnn=[samples['acnn_{}_{}'.format(
                        ii, _set)].get_fname() for ii in xrange(num_attn_cnn)],
                    fname_dcnn=[samples['dcnn_{}_{}'.format(
                        ii, _set)].get_fname() for ii in xrange(num_dcnn)])

                if not samples['output_{}'.format(_set)].is_registered():
                    for _name in ['input', 'output', 'total', 'box', 'patch']:
                        samples['{}_{}'.format(_name, _set)].register()
                    for _name, _num in zip(
                            ['ccnn', 'acnn', 'dcnn'],
                            [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                        [samples['{}_{}_{}'.format(_name, ii, _set)].register()
                         for ii in xrange(_num)]
        pass

    def run_validation(step):
        """Validation"""
        loss = 0.0
        conf_loss = 0.0
        segm_loss = 0.0
        box_loss = 0.0
        iou_soft = 0.0
        iou_hard = 0.0
        count_acc = 0.0
        dice = 0
        dic = 0
        dic_abs = 0
        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])
        bn = {}

        log.info('Running validation')
        for _x, _y, _s in BatchIterator(num_ex_valid,
                                        batch_size=batch_size,
                                        get_fn=get_batch_valid,
                                        progress_bar=False):
            results_list = [m['loss'], m['conf_loss'], m['segm_loss'],
                            m['box_loss'], m['iou_soft'], m['iou_hard'],
                            m['count_acc'], m['dice'], m['dic'],
                            m['dic_abs']]
    
            offset = len(results_list)

            for _layer, _num in zip(['ctrl_cnn', 'attn_cnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                for ii in xrange(_num):
                    for tt in xrange(timespan):
                        for _stat in ['bm', 'bv', 'em', 'ev']:
                            results_list.append(
                                m['{}_{}_{}_{}'.format(_layer, ii, _stat, tt)])

            results = sess.run(results_list,
                               feed_dict={
                                   m['x']: _x,
                                   m['phase_train']: False,
                                   m['y_gt']: _y,
                                   m['s_gt']: _s
                               })
            _loss = results[0]
            _conf_loss = results[1]
            _segm_loss = results[2]
            _box_loss = results[3]
            _iou_soft = results[4]
            _iou_hard = results[5]
            _count_acc = results[6]
            _dice = results[7]
            _dic = results[8]
            _dic_abs = results[9]
            _bn = {}

            num_ex_batch = _x.shape[0]

            # Batch normalization stats.
            for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                for ii in xrange(_num):
                    for tt in xrange(timespan):
                        for _stat in ['bm', 'bv', 'em', 'ev']:
                            key = '{}_{}_{}_{}'.format(_layer, ii, _stat, tt)
                            delta = results[offset] * num_ex_batch / \
                                num_ex_valid
                            if key in bn:
                                bn[key] += delta
                            else:
                                bn[key] = delta
                            offset += 1

            loss += _loss * num_ex_batch / num_ex_valid
            conf_loss += _conf_loss * num_ex_batch / num_ex_valid
            segm_loss += _segm_loss * num_ex_batch / num_ex_valid
            box_loss += _box_loss * num_ex_batch / num_ex_valid
            iou_soft += _iou_soft * num_ex_batch / num_ex_valid
            iou_hard += _iou_hard * num_ex_batch / num_ex_valid
            count_acc += _count_acc * num_ex_batch / num_ex_valid
            dice += _dice * num_ex_batch / num_ex_valid
            dic += _dic * num_ex_batch / num_ex_valid
            dic_abs += _dic_abs * num_ex_batch / num_ex_valid

        log.info(('{:d} vtl {:.4f} cl {:.4f} sl {:.4f} bl {:.4f} '
                  'ious {:.4f} iouh {:.4f} dice {:.4f}').format(
            step, loss, conf_loss, segm_loss, box_loss, iou_soft, iou_hard,
            dice))

        if args.logs:
            loss_logger.add(step, ['', loss])
            conf_loss_logger.add(step, ['', conf_loss])
            segm_loss_logger.add(step, ['', segm_loss])
            box_loss_logger.add(step, ['', box_loss])
            iou_logger.add(step, ['', iou_soft, '', iou_hard])
            dice_logger.add(step, ['', dice])
            count_acc_logger.add(step, ['', count_acc])
            dic_logger.add(step, ['', dic])
            dic_abs_logger.add(step, ['', dic_abs])

            # Batch normalization stats.
            for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                for ii in xrange(_num):
                    for tt in xrange(timespan):
                        _prefix = '{}_{}_{{}}_{}'.format(_layer, ii, tt)
                        _output = ['', bn[_prefix.format('bm')],
                                   '', bn[_prefix.format('bv')], '', '']
                        bn_loggers['{}_{}_{}'.format(
                            _layer, ii, tt)].add(step, _output)

        pass

    def train_step(step, x, y, s):
        """Train step"""
        start_time = time.time()

        num_ctrl_cnn = len(model_opt['ctrl_cnn_filter_size'])
        num_attn_cnn = len(model_opt['attn_cnn_filter_size'])
        num_dcnn = len(model_opt['dcnn_filter_size'])
        attn_cnn_weights_m = [0.0] * num_attn_cnn
        attn_cnn_bias_m = [0.0] * num_attn_cnn
        sparsity = {}

        results_list = [m['loss'], m['conf_loss'], m['segm_loss'], m['box_loss'],
                        m['iou_soft'], m['iou_hard'], m['learn_rate'],
                        m['crnn_g_i_avg'], m['crnn_g_f_avg'], m['crnn_g_o_avg'],
                        m['box_loss_coeff'], m['count_acc'],
                        m['gt_knob_prob_box'], m['gt_knob_prob_segm'],
                        m['dice'], m['dic'], m['dic_abs'], m['attn_lg_gamma'],
                        m['attn_lg_var'], m['attn_box_lg_gamma'],
                        m['y_out_lg_gamma'], m['attn_box_beta'],
                        m['y_out_beta']]

        offset = len(results_list)

        # Batch normalization
        for _layer, _num in zip(['ctrl_cnn', 'attn_cnn', 'dcnn'],
                                [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
            for ii in xrange(_num):
                for tt in xrange(timespan):
                    for _stat in ['bm', 'bv', 'em', 'ev']:
                        results_list.append(
                            m['{}_{}_{}_{}'.format(_layer, ii, _stat, tt)])

        results_list.extend(m['acnn_w_mean'])
        results_list.extend(m['acnn_b_mean'])
        results_list.extend(m['h_ccnn'])
        results_list.extend(m['h_acnn'])
        results_list.extend(m['h_dcnn'])

        results_list.append(m['train_step'])

        results = sess.run(results_list,
                           feed_dict={
                               m['x']: x,
                               m['phase_train']: True,
                               m['y_gt']: y,
                               m['s_gt']: s
                           })

        # Print statistics
        if step % train_opt['steps_per_log'] == 0:
            loss = results[0]
            conf_loss = results[1]
            segm_loss = results[2]
            box_loss = results[3]
            iou_soft = results[4]
            iou_hard = results[5]
            learn_rate = results[6]
            crnn_g_i_avg = results[7]
            crnn_g_f_avg = results[8]
            crnn_g_o_avg = results[9]
            box_loss_coeff = results[10]
            count_acc = results[11]
            gt_knob_box = results[12]
            gt_knob_segm = results[13]
            dice = results[14]
            dic = results[15]
            dic_abs = results[16]
            attn_lg_gamma = results[17]
            attn_lg_var = results[18]
            attn_box_lg_gamma = results[19]
            y_out_lg_gamma = results[20]
            attn_box_beta = results[21]
            y_out_beta = results[22]

            # Batch normalization stats.
            bn = {}
            for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                for ii in xrange(_num):
                    for tt in xrange(timespan):
                        for _stat in ['bm', 'bv', 'em', 'ev']:
                            bn['{}_{}_{}_{}'.format(_layer, ii, _stat, tt)] = \
                                results[offset]
                            offset += 1

            for ii in xrange(num_attn_cnn):
                attn_cnn_weights_m[ii] = results[offset]
                offset += 1

            for ii in xrange(num_attn_cnn):
                attn_cnn_bias_m[ii] = results[offset]
                offset += 1

            # Layer sparsity
            for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                    [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                sparsity[_layer] = []
                for ii in xrange(_num):
                    sparsity[_layer].append(
                        (results[offset] == 0).astype('float').mean())
                    offset += 1

            step_time = (time.time() - start_time) * 1000
            log.info(('{:d} tl {:.4f} cl {:.4f} sl {:.4f} bl {:.4f} '
                      'ious {:.4f} iouh {:.4f} dice {:.4f} t {:.2f}ms').format(
                step, loss, conf_loss, segm_loss, box_loss, iou_soft, iou_hard,
                dice, step_time))

            if args.logs:
                loss_logger.add(step, [loss, ''])
                conf_loss_logger.add(step, [conf_loss, ''])
                segm_loss_logger.add(step, [segm_loss, ''])
                box_loss_logger.add(step, [box_loss, ''])
                iou_logger.add(step, [iou_soft, '', iou_hard, ''])
                dice_logger.add(step, [dice, ''])
                count_acc_logger.add(step, [count_acc, ''])
                dic_logger.add(step, [dic, ''])
                dic_abs_logger.add(step, [dic_abs, ''])
                learn_rate_logger.add(step, learn_rate)
                box_loss_coeff_logger.add(step, box_loss_coeff)
                step_time_logger.add(step, step_time)
                crnn_logger.add(
                    step, [crnn_g_i_avg, crnn_g_f_avg, crnn_g_o_avg])
                gt_knob_logger.add(step, [gt_knob_box, gt_knob_segm])

                # Batch normalization stats.
                for _layer, _num in zip(['ccnn', 'acnn', 'dcnn'],
                                        [num_ctrl_cnn, num_attn_cnn, num_dcnn]):
                    for ii in xrange(_num):
                        for tt in xrange(timespan):
                            _prefix = '{}_{}_{{}}_{}'.format(_layer, ii, tt)
                            _output = [bn[_prefix.format('bm')], '',
                                       bn[_prefix.format('bv')], '',
                                       bn[_prefix.format('em')],
                                       bn[_prefix.format('ev')]]
                            bn_loggers['{}_{}_{}'.format(
                                _layer, ii, tt)].add(step, _output)

                # Attn parameters
                attn_params_logger.add(
                    step, [attn_lg_gamma.mean(), attn_lg_var.mean(),
                           attn_box_lg_gamma.mean(),
                           y_out_lg_gamma.mean(), attn_box_beta.mean(),
                           y_out_beta.mean()])

                # Weights
                acnn_weights_stats = []
                for ii in xrange(num_attn_cnn):
                    acnn_weights_stats.append(attn_cnn_weights_m[ii])
                    acnn_weights_stats.append(attn_cnn_bias_m[ii])
                acnn_weights_logger.add(step, acnn_weights_stats)

                # Layer sparsity
                for _layer in ['ccnn', 'acnn', 'dcnn']:
                    sparsity_loggers[_layer].add(step, sparsity[_layer])

        pass

    def train_loop(step=0):
        """Train loop"""
        for x_bat, y_bat, s_bat in BatchIterator(num_ex_train,
                                                 batch_size=batch_size,
                                                 get_fn=get_batch_train,
                                                 cycle=True,
                                                 progress_bar=False):
            # Run validation
            if step % train_opt['steps_per_valid'] == 0:
                run_validation(step)

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
    learn_rate_logger.close()
    step_time_logger.close()

    pass
