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
    file_list = [f.strip() for f in file_list]
    if check:
        log.info('Checking all files exist')
        N = len(file_list)
        pb = progress_bar.get_iter(file_list)
        for i, f in enumerate(pb):
            if not os.path.exists(f):
                log.fatal('File not found: {0}'.format(f))
    return file_list
