import numpy as np
import os
import sharded_hdf5 as sh
import unittest


class ShardedFileTests(unittest.TestCase):
    """Unit tests for ShardedFile, ShardedFileReader, and ShardedFileWriter."""

    def test_read_write(self):
        N = 100
        N1 = 10
        N2 = 8
        D1 = 10
        D2 = 5
        num_shards = 10

        f = sh.ShardedFile('test', num_shards=num_shards)

        with sh.ShardedFileWriter(f, num_objects=N) as writer:
            for i in writer:
                data = {
                    'key1': np.zeros((N1, D1)) + i,
                    'key2': np.zeros((N2, D2)) - i
                }
                writer.write(data)

        with sh.ShardedFileReader(f) as reader:
            for i, data in enumerate(reader):
                self.assertTrue((data['key1'] == i).all())
                self.assertTrue((data['key2'] == -i).all())

        with sh.ShardedFileReader(f, batch_size=3) as reader:
            for i, data in enumerate(reader):
                dkey1 = np.concatenate([d['key1'] for d in data], axis=0)
                dkey2 = np.concatenate([d['key2'] for d in data], axis=0)
                if dkey1.shape[0] == 3 * N2:
                    self.assertTrue((dkey1[: N1] == 3 * i).all())
                    self.assertTrue((dkey1[N1: 2 * N2] == 3 * i + 1).all())
                    self.assertTrue((dkey1[2 * N1:] == 3 * i + 2).all())
                    self.assertTrue((dkey2[: N2] == -3 * i).all())
                    self.assertTrue((dkey2[N2: 2 * N2] == -3 * i - 1).all())
                    self.assertTrue((dkey2[2 * N2:] == -3 * i - 2).all())
                elif dkey1.shape[0] == 2 * N2:
                    self.assertTrue((dkey1[: N1] == 3 * i).all())
                    self.assertTrue((dkey1[N1: 2 * N2] == 3 * i + 1).all())
                    self.assertTrue((dkey2[: N2] == -3 * i).all())
                    self.assertTrue((dkey2[N2: 2 * N2] == -3 * i - 1).all())
                elif dkey1.shape[0] == N2:
                    self.assertTrue((dkey1 == 3 * i).all())
                    self.assertTrue((dkey2 == -3 * i).all())

        for i in xrange(num_shards):
            os.remove(f.get_fname(i))

        pass

    def test_read_keys(self):
        N = 100
        N1 = 10
        D1 = 10
        num_shards = 10

        f = sh.ShardedFile('test2', num_shards=num_shards)

        with sh.ShardedFileWriter(f, num_objects=N) as writer:
            for i in writer:
                data = {
                    'key': i,
                    'value': np.zeros((N1, D1)) + i
                }
                writer.write(data)

        with sh.ShardedFileReader(f, key_name='key') as reader:
            for i in xrange(N):
                data = reader[i]
                self.assertTrue((data['value'] == i).all())
                self.assertTrue(data['key'] == i)

        for i in xrange(num_shards):
            os.remove(f.get_fname(i))

        pass

if __name__ == '__main__':
    unittest.main()
