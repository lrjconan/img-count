import caffe
from utils import logger

log = logger.get()


class FastRCNN_Model(object):

    def __init__(self, name, model_def_fname, weights_fname,
                 gpu=False, gpu_id=0):
        self.name = name
        self.model_def_fname = model_def_fname
        self.weights_fname = weights_fname
        if gpu:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
            log.info('Using GPU {:d}'.format(gpu_id))
        else:
            caffe.set_mode_cpu()
            log.info('Using CPU')
        self.net = caffe.Net(self.model_def_fname, self.weights_fname,
                             caffe.TEST)
        print '\n\nLoaded network {}{}'.format(self.name, self.weights_fname)
        pass
    pass
