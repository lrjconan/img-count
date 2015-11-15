class Dataset(object):
    """
    Category dictionary
    Vocabulary dictionary
    Image IDs
    """
    
    def get_cat_dict(self):
        raise Exception('Not implemented')
    
    def get_cat_list(self):
        # Add background class to be consistent with Fast-RCNN encoding.
        cat_dict = self.get_cat_dict()
        cat_list = ['__background__']
        for cat in cat_dict.itervalues():
            cat_list.append(cat)
        return cat_list

    pass
