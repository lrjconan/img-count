from utils.sharded_hdf5 import ShardedFile
import h5py

def add_key(fname, num_shards, key_name):
    f = ShardedFile(fname, num_shards)
    for i in xrange(num_shards):
        fname_i = f.get_fname(i)
        fh = h5py.File(fname_i)
        fh['__keys__'] = fh[key_name][:]

# def add_remove_key(fname, num_shards, key_name):
#     f = ShardedFile(fname, num_shards)
#     for i in xrange(num_shards):
#         fname_i = f.get_fname(i)
#         fh = h5py.File(fname_i)
#         fh['__keys__'] = fh[key_name][:]
#         fh.del(key_name)
#         fh.del('__sep_{}__'.format(key_name))
