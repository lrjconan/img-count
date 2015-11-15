import logger
import os.path
import progress_bar

log = logger.get()

def read_file_list(fname, check=False):
    """
    Reads file names in a plain text file.

    Args:
        fname: string, path to line delimited file names.
        check: bool, check whether the files all exist.
    Returns:
        file_list: list, list of file names in string.
    Raises:
        Exception: image file name does not exists.
    """
    file_list = list(open(fname))
    for i, f in enumerate(file_list):
        file_list[i] = f.strip()
    if check:
        log.info('Checking all files exist')
        N = len(file_list)
        pb = progress_bar.get(N)
        for i, f in enumerate(file_list):
            if not os.path.exists(f):
                raise Exception('File not found: {0}'.format(f))
            pb.increment()
    return file_list
