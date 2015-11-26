"""
COCO-Count Dataset API.
"""

import os.path


class COCOCount(object):

    def __init__(base_dir, set_name='train', feat_layer='pool5',
                 groundtruth=False):
        if set_name != 'train' and set_name != 'valid':
            raise Exception('Set name {} not found'.format(set_name))
        pass
        self._base_dir = base_dir
        self._set_name = set_name

        self._groundtruth = groundtruth
        self._feat_layer = feat_layer
        self._info_data_path = os.path.join(
            self.base_dir, 'coco_count_info_{}-*'.format(set_name))

    def get_info_data_path():
        return self._info_data_path
        pass

    def get_train_data():
        pass

    def get_valid_data():
        pass
    pass
