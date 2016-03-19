import cslab_environ

import tensorflow as tf

from utils import logger
from utils.saver import Saver
import fg_segm_models as models
import h5py

log = logger.get()


def read(folder):
    log.info('Reading foreground segmentation network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    model = models.get_model(model_opt)
    cnn_nlayers = len(model_opt['cnn_filter_size'])
    weights = {}
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)
    for ii in xrange(cnn_nlayers):
        w_name = 'cnn_w_{}'.format(ii)
        b_name = 'cnn_b_{}'.format(ii)
        weights[w_name] = sess.run(model[w_name])
        weights[b_name] = sess.run(model[b_name])
        log.info(w_name)
        log.info(weights[w_name].shape)
        log.info(b_name)
        log.info(weights[b_name].shape)

    return weights


def save(fname, folder):
    weights = read(folder)
    h5f = h5py.File(fname, 'w')
    for key in weights:
        h5f[key] = weights[key]
    h5f.close()

    pass

if __name__ == '__main__':
    save('/ais/gobi3/u/mren/results/img-count/fg_segm-20160318131644/weights.h5',
         '/ais/gobi3/u/mren/results/img-count/fg_segm-20160318131644')
