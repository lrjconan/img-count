import tensorflow as tf

from utils import logger
from utils.saver import Saver
import fg_segm_models as models

log = logger.get()


def read(folder):
    log.info('Reading foreground segmentation network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    model = models.get_model(model_opt)
    cnn_nlayers = model_opt['cnn_filter_size']
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

    pass

if name == '__main__':
    read('/ais/gobi3/u/mren/results/img-count/fg_segm-20160318131751')
