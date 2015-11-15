from dataset import Dataset
from pycocotools.coco import COCO
import os.path

train_annotation_fn = 'annotations/instances_train2014.json'
valid_annotation_fn = 'annotations/instances_val2014.json'


class Mscoco(Dataset):
    """
    MS-COCO API
    """
    def __init__(self, base_dir, set_name='train'):
        if set_name == 'train':
            self._annotation_fn = os.path.join(base_dir, train_annotation_fn)
        elif set_name == 'valid':
            self._annotation_fn = os.path.join(base_dir, valid_annotation_fn)
        else:
            raise Exception('Set name {} not found'.format(set_name))
        self._coco = COCO(self._annotation_fn)
        pass

    def get_cat_dict(self):
        cat_dict = {}
        for cat_id in self._coco.cats.iterkeys():
            cat_dict[cat_id] = self._coco.cats[cat_id]['name']
        return cat_dict
