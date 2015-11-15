import caffe


class FastRCNNModel(object):

    def __init__(self, name, model_def_fname, weights_fname, gpu=False):
        self.name = name
        self.model_def_fname = model_def_fname
        self.weights_fname = weights_fname
        if gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(self.model_def_fname, self.weights_fname, 
                             caffe.TEST)
        print '\n\nLoaded network {}{}'.format(self.name, self.weights_fname)
        pass
    pass
