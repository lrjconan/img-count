"""
Sharded HDF5 format: storing a bundle of sharded files.

==File system structure:
    1. File pattern
    * /path/folder/file_prefix-{index}-{num_shards}{suffix}
    * {index} and {num_shards} are 5 digit 0-padded integer string.
    * e.g. /path/folder/example-00010-of-00020.h5

    2. HDF5 structure
    {
        'num_items__': int, number of items in this file.
        'sep__': 1D separator int64 array for each item.
        'key1': 2D array.
        'key2': 2D array.
        ...
    }

    3. Object mapping
    {
        'key1': 1D or 2D array.
        'key2': 1D or 2D array.
        ...
    }

==Key classes:
    1. ShardedFile
    2. ShardedFileReader
    3. ShardedFileWriter

==Example usages:
    1. Read: iterate everything in batch
    >> with ShardedFileReader(ShardedFile('a', 100), batch_size=10) as reader:
    >>    for items in reader:
    >>        do(items)

    2. Read: iterate from a position
    >> with ShardedFileReader(ShardedFile('a', 100), batch_size=10) as reader:
    >>     for items in reader.seek(50):
    >>         do(items)

    3. Read: random access
    >> with ShardedFileReader(ShardedFile('a', 100)) as reader:
    >>     reader.seek(pos=position)
    >>     data = reader.read(num_items=100)

    4.  Write in order
    >> with ShardedFileWriter(ShardedFile('a', 10), num_objects=100) as writer:
    >>     for i in writer:
    >>         writer.write(data[i])
"""


import bisect
import h5py
import logger
import math
import numpy
import os
import re

log = logger.get()

ERR_MSG_IDX_TOO_LARGE = 'Shard index larger than number of shards: {}'
ERR_MSG_IDX_TOO_LARGE2 = 'Shard index larger than number of shards'
ERR_MSG_MISSING_FILE = 'Missing file for index {:d} out of {:d}'
ERR_MSG_DUPE_FILES = 'Duplicate files found for index {:d}'
ERR_MSG_MISSING_NUM_ITEMS_FIELD = 'Missing field "num_items" in file {}'
KEY_NUM_ITEM = '__num_items__'
KEY_SEPARATOR = '__sep__'


class ShardedFile(object):
    """Sharded file object.
    """

    def __init__(self, file_prefix, num_shards, suffix='.h5'):
        """Construct a sharded file instance.
        """
        if not isinstance(num_shards, int):
            raise Exception('Number of shards need to be integer')
        self.file_prefix = os.path.abspath(file_prefix)
        self.basename = os.path.basename(self.file_prefix)
        log.info(self.basename)
        self.num_shards = num_shards
        self.suffix = suffix

    def get_fname(self, shard):
        """Get the file name for a specific shard.

        Args:
            shard: int, shard index.
        """
        if shard >= self.num_shards:
            raise Exception(ERR_MSG_IDX_TOO_LARGE2)

        return '{}-{:05d}-of-{:05d}{}'.format(
            self.file_prefix, shard, self.num_shards, self.suffix)


class ShardedFileReader(object):
    """Shareded file reader.
    """

    def __init__(self, sharded_file, batch_size=1, check=True):
        """Construct a sharded file reader instance.

        Args:
            sharded_file: SharededFile instance.
            batch_size: number, average batch_size for each read. The actual 
            size depends on the number of items in a file so is not guaranteed 
            to be the same size.
        """
        self.file = sharded_file

        # Batch size of reading.
        self.batch_size = batch_size

        # Position index in each file for binary search.
        self._file_index = None

        # Reader position.
        self._pos = 0

        # Current file ID.
        self._cur_fid = 0

        # Current file handler.
        self._fh = None

        # Whether need to refresh file handler in the next read.
        self._need_refresh = False

        # Current file separator.
        self._cur_sep = []

        # Check files all exist.
        if check:
            self._check_files()

        pass

    def __iter__(self):
        """Get an iterator.
        """
        return self

    def __enter__(self):
        """Enter with clause.
        """
        return self

    def __exit__(self, type, value, traceback):
        """Exit with clause.
        """
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        pass

    def _check_files(self):
        """Check existing files"""

        fname_re = re.compile(
            '({})-([0-9]{{5}})-of-{:05d}{}'.format(
                self.file.basename,
                self.file.num_shards,
                self.file.suffix))
        dirname = os.path.dirname(os.path.abspath(self.file.file_prefix))
        files_in_dir = os.listdir(dirname)
        found_files = {}

        # Look for files in the expected pattern.
        for fname in files_in_dir:
            match = fname_re.match(fname)
            if match is not None:
                index_str = match.groups()[1]
                index_int = int(index_str)
                if index_int >= self.file.num_shards:
                    raise Exception(ERR_MSG_IDX_TOO_LARGE.format(fname))
                if index_int in found_files:
                    raise Exception(ERR_MSG_DUPE_FILES.format(index_int))
                found_files[index_int] = fname

        # Check all files in the sequence exist.
        for idx in xrange(self.file.num_shards):
            if idx not in found_files:
                raise Exception(ERR_MSG_MISSING_FILE.format(idx, total))

        log.info('Check file success: {}'.format(self.file.file_prefix))

        pass

    def _build_index(self):
        """Build a mapping from an index to shard number.

        Returns:
            file_index: list, end element id - 1 of each shard.
        """
        log.info('Building index')
        file_index = []
        index = 0
        for shard_idx in xrange(self.file.num_shards):
            fname = self.file.get_fname(shard_idx)
            fh = h5py.File(fname, 'r')
            if KEY_NUM_ITEM in fh:
                num_items = fh[KEY_NUM_ITEM][0]
            else:
                raise Exception(ERR_MSG_MISSING_NUM_ITEMS_FIELD.format(fname))
            index += num_items
            file_index.append(index)

        return file_index

    def find(self, index):
        """Find the file id.

        Args:
            index: number, item index.

        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()
        return bisect.bisect_left(self._file_index, index)

    def read(self, num_items=1):
        """Read from the current position.

        Args:
            num_items: number, number of desired items to read. It is not 
            guaranteed to return the exact same number of items
        Returns:
            results: dict, keys are same with the keys defined in the file,
            values are numpy.ndarray.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        # Renew file ID.
        if self._need_refresh:
            self._cur_fid += 1
            if self._fh is not None:
                self._fh.close()
            self._fh = h5py.File(self.file.get_fname(self._cur_fid))
            self._need_refresh = False
            self._cur_sep = self._fh[KEY_SEPARATOR][:]

        # Open a file.
        if self._fh is None:
            self._fh = h5py.File(self.file.get_fname(self._cur_fid))
            self._cur_sep = self._fh[KEY_SEPARATOR][:]

        # Compute file_start and file_end (absolute cursor) and 
        # item_start and item_end (relative cursor).
        if self._cur_fid == 0:
            file_start = 0
        else:
            file_start = self._file_index[self._cur_fid - 1]
        file_end = self._file_index[self._cur_fid]

        item_start = self._pos - file_start
        item_end = min(self._pos + num_items, file_end) - file_start

        # Refresh next time if reached the end.
        if item_end == file_end - file_start:
            self._need_refresh = True

        # Compute line start and end.
        if item_start == 0:
            line_start = 0
        else:
            line_start = self._cur_sep[item_start - 1]
        line_end = self._cur_sep[item_end - 1]

        # Read data.
        results = {}
        for key in self._fh.keys():
            if not key.startswith('__'):
                results[key] = self._fh[key][line_start: line_end]

        self._pos += num_items

        return results

    def seek(self, pos):
        """Seek to specific position.

        Args:
            pos: number, position in terms of number of items.
        Returns:
            A SharededReader instance.
        """
        self._pos = pos
        fid = self.find(self._pos)
        if fid != self._cur_fid or self._fh is None:
            self._cur_fid = fid
            if self._fh is not None:
                self._fh.close()
            self._fh = h5py.File(self.file.get_fname(fid))

        return self

    def next(self):
        """Iterate to next batch to read.
        """
        # Lazy build file index.
        if self._file_index is None:
            self._file_index = self._build_index()

        if self._pos < self._file_index[-1]:
            return self.read(self.batch_size)
        else:
            raise StopIteration()

    def close(self):
        self.__exit__(None, None, None)


class ShardedFileWriter(object):
    """Sharded file writer.
    """

    def __init__(self, sharded_file, num_objects):
        """Construct a sharded file writer instance.

        Args:
            sharded_file: ShardedFile instance.
            num_objects: number, total number of objects to write.
        """
        self.file = sharded_file

        # Total number of items to write.
        self._num_objects = num_objects

        # Total number of shards.
        self._num_shards = self.file.num_shards

        # Number of items per shard.
        self._num_objects_per_shard = int(
            math.ceil(num_objects / float(self._num_shards)))

        # Current file handler.
        self._fh = h5py.File(self.file.get_fname(0), 'w')

        # Current item index.
        self._pos = 0

        # Current shard index.
        self._shard = 0

        # File buffer for current shard.
        self._buffer = {}

        # Number of items for current shard.
        self._cur_num_items = 0

        # Index separator for current shard.
        self._cur_sep = []

        pass

    def __enter__(self):
        """Enter with clause.
        """

        return self

    def __exit__(self, type, value, traceback):
        """Exit with clause.
        """

        self._flush()
        if self._fh is not None:
            self._fh.close()

        pass

    def __iter__(self):
        """Get an iterator.
        """
        
        return self

    def write(self, data):
        """Write a single entry into buffer.
        """

        # Check that all entries has the same first dimension.
        # Check that all entries have the same set of keys.
        shape1 = None
        for key in data.iterkeys():
            if shape1 is None:
                shape1 = data[key].shape[0]
            else:
                if data[key].shape[0] != shape1:
                    raise Exception('First dimension does not match.')
            if len(self._buffer) > 0:
                if key not in self._buffer:
                    raise Exception('Unknown key: {}'.format(key))

        for key in data.iterkeys():
            if key in self._buffer:
                self._buffer[key].append(data[key])
            else:
                self._buffer[key] = [data[key]]

        self._cur_num_items += 1
        if len(self._cur_sep) > 0:
            last = self._cur_sep[-1]
            self._cur_sep.append(last + shape1)
        else:
            self._cur_sep.append(shape1)

        pass

    def _flush(self):
        """Flush the buffer into the current shard.
        """
        if len(self._buffer) > 0:
            for key in self._buffer.iterkeys():
                value = numpy.concatenate(self._buffer[key], axis=0)
                print key, value
                self._fh[key] = value
            self._fh[KEY_NUM_ITEM] = numpy.array([self._cur_num_items])
            self._fh[KEY_SEPARATOR] = numpy.array(self._cur_sep, dtype='int64')
            self._cur_num_items = 0
            self._cur_sep = []
            self._buffer = {}

        pass

    def next_file(self):
        """Move to writing the next shard.
        """
        if self._cur_num_items > 0:
            self._flush()

        self._shard += 1
        if self._shard >= self.file.num_shards:
            raise Exception(ERR_MSG_IDX_TOO_LARGE2)
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._fh = h5py.File(self.file.get_fname(self._shard), 'w')

        pass

    def next(self):
        """Move to writing the next object.
        """

        if self._pos < self._num_objects:
            r = self._pos - self._shard * self._num_objects_per_shard
            if r == self._num_objects_per_shard:
                self.next_file()
            i = self._pos
            self._pos += 1
            return i
        else:
            raise StopIteration()
        pass

    def close(self):
        """Close the opened file.
        """

        self.__exit__(None, None, None)
