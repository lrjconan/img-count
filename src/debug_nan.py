import cslab_environ

from data_api import cvppp
from utils.saver import Saver
from utils import logger
import numpy as np
import ris_double_attn_model as double_attention
import ris_attn_model as attention
import sys
import os
import tensorflow as tf
import h5py


log = logger.get()

if __name__ == '__main__':
    model_folder = sys.argv[1]
    data_folder = '/ais/gobi3/u/mren/data/lsc/A1'

    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()

    ckpt_fname = ckpt_info['ckpt_fname']
    step = ckpt_info['step']

    model_opt = ckpt_info['model_opt']
    data_opt = ckpt_info['data_opt']
    if 'squash_ctrl_params' not in model_opt:
        model_opt['squash_ctrl_params'] = False
    if 'use_iou_box' not in model_opt:
        model_opt['use_iou_box'] = True
    if 'clip_gradient' not in model_opt:
        model_opt['clip_gradient'] = 1.0

    if model_opt['type'] == 'double_attention':
        model = double_attention.get_model(model_opt)
    elif model_opt['type'] == 'attention':
        model = attention.get_model(model_opt)

#############
# Run data 
#############
    data = h5py.File(os.path.join(model_folder, 'nan_input.h5'), 'r')
    x = data['x'][:]
    y = data['y'][:]
    s = data['s'][:]

    # all_data = cvppp.get_dataset(data_folder, data_opt)
    # dataset = {}
    # split = 103
    # random = np.random.RandomState(2)
    # idx = np.arange(all_data['input'].shape[0])
    # random.shuffle(idx)
    # train_idx = idx[: split]
    # valid_idx = idx[split:]
    # log.info('Train index: {}'.format(train_idx))
    # log.info('Valid index: {}'.format(valid_idx))
    # dataset['train'] = {
    #     'input': all_data['input'][train_idx],
    #     'label_segmentation': all_data['label_segmentation'][train_idx],
    #     'label_score': all_data['label_score'][train_idx]
    # }

    # batch_start = (step * 8) % train_idx.size
    # batch_end = max(batch_start + 8, train_idx.size)
    # x_bat = dataset['train']['input'][batch_start: batch_end]
    # y_bat = dataset['train']['label_segmentation'][batch_start: batch_end]
    # s_bat = dataset['train']['label_score'][batch_start: batch_end]

    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    # loss = sess.run(model['loss'], feed_dict={model['x']: x_bat,
    #                                           model['y_gt']: y_bat,
    #                                           model['s_gt']: s_bat,
    #                                           model['phase_train']: True})
    # log.info('loss: {:6.4f}'.format(loss))

    model_val = {}
    output_list = []

    for cnn in ['ctrl_cnn', 'attn_cnn', 'attn_dcnn']:
        # for cnn in ['ctrl_cnn']:
        for ii in xrange(len(model_opt['{}_filter_size'.format(cnn)])):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(cnn, w, ii)
                log.info(key)
                output_list.append(key)

    for lstm in ['ctrl_lstm']:
        for comp in ['i', 'f', 'i', 'o']:
            for w in ['w_x', 'w_h', 'b_']:
                key = '{}_{}{}'.format(lstm, w, comp)
                log.info(key)
                output_list.append(key)

    if model_opt['type'] == 'double_attention':
        mlps = ['ctrl_mlp', 'glimpse_mlp', 'score_mlp', 'attn_mlp']
        mlplayers = [model_opt['num_ctrl_mlp_layers'],
                     model_opt['num_glimpse_mlp_layers'],
                     1,
                     model_opt['num_attn_mlp_layers']]
    elif model_opt['type'] == 'attention':
        mlps = ['ctrl_mlp', 'score_mlp', 'attn_mlp']
        mlplayers = [model_opt['num_ctrl_mlp_layers'],
                     1,
                     model_opt['num_attn_mlp_layers']]

    for mlp, nlayers in zip(mlps, mlplayers):
        for ii in xrange(nlayers):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(mlp, w, ii)
                log.info(key)
                output_list.append(key)

    output_var = []
    for key in output_list:
        output_var.append(model[key])

    output_val = sess.run(output_var)
    for kk, key in enumerate(output_list):
        model_val[key] = output_val[kk]

    for key in model_val.iterkeys():
        val = model_val[key]
        log.info('Is NaN   {:15s} {}'.format(
            key, val.size == np.isnan(val).sum()))

    for key in model_val.iterkeys():
        val = model_val[key]
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Mean', key, val.mean(), np.abs(val).mean()))
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Max', key, val.max(), np.abs(val).max()))
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Min', key, val.min(), np.abs(val).min()))

    print 'ctrl_mlp_b_0'
    print model_val['ctrl_mlp_b_0']
    print 'ctrl_mlp_w_0'
    print model_val['ctrl_mlp_w_0']
    print 'ctrl_cnn_w_0'
    print model_val['ctrl_cnn_w_0']
