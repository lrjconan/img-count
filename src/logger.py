from __future__ import print_function
from termcolor import colored
import datetime
import inspect
import os

terminal = {
    'normal': '\033[0m',
    'bright': '\033[1m',
    'invert': '\033[7m',
    'black': '\033[30m', 
    'red': '\033[31m', 
    'green':'\033[32m', 
    'yellow': '\033[33m', 
    'blue': '\033[34m', 
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m', 
    'default': '\033[39m'
}

def get(filename=None):
    return Logger(filename)

class Logger(object):

    def __init__(self, filename=None):
        now = datetime.datetime.now()
        self.verbose_thresh = os.environ.get('VERBOSE', 0)
        if filename is not None:
            self.filename = \
                '{}-{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}.log'.format(
                filename, 
                now.year, now.month, now.day, now.hour, now.minute, now.second)
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            open(self.filename, 'w').close()
        else:
            self.filename = None
        pass
    
    @staticmethod
    def get_time_str(t=datetime.datetime.now()):
        timestr = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
                t.year, t.month, t.day, t.hour, t.minute, t.second)
        return timestr

    def log(self, message, typ='info', verbose=0):
        if typ == 'info':
            typstr_print = '{0}INFO:{1}'.format(
                    terminal['green'], terminal['default'])
            typstr_log = 'INFO:'
        elif typ == 'warning':
            typstr_print = '{0}WARNING:{1}'.format(
                    terminal['yellow'], terminal['default'])
            typstr_log = 'WARNING'
        elif typ == 'error':
            typstr_print = '{0}ERROR:{1}'.format(
                    terminal['red'], terminal['default'])
            typstr_log = 'ERROR'
        elif typ == 'fatal':
            typstr_print = '{0}FATAL:{1}'.format(
                    terminal['red'], terminal['default'])
            typstr_log = 'FATAL'
        else:
            raise Exception('Unknown log type: {0}'.format(typ));
        timestr = self.get_time_str()
        for (frame, filename, line_number, function_name, lines, index) in \
                inspect.getouterframes(inspect.currentframe()):
            if not filename.endswith('logger.py'):
                break
        cwd = os.getcwd()
        if filename.startswith(cwd): filename = filename[len(cwd):]
        filename = filename.lstrip('/')
        callerstr = '{0}:{1}'.format(filename, line_number)
        printstr = '{0} {1} {2} {3}'.format(
                typstr_print, timestr, callerstr, message)
        logstr = '{0} {1} {2} {3}'.format(
                typstr_log, timestr, callerstr, message)
        if self.verbose_thresh <= verbose:
            print(printstr)
        if self.filename is not None:
            with open(self.filename, 'a') as f:
                f.write(logstr)
                f.write('\n')
        pass

    def info(self, message, verbose=0):
        self.log(message, typ='info', verbose=verbose)
        pass

    def warning(self, message, verbose=0):
        self.log(message, typ='warning', verbose=verbose)
        pass

    def error(self, message, verbose=0):
        self.log(message, typ='error', verbose=verbose)
        pass

    def fatal(self, message, verbose=0):
        self.log(message, typ='fatal', verbose=verbose)
        sys.exit(0)
        pass

