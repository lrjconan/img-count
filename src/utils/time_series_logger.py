import datetime
import logger
import os

log = logger.get()


class TimeSeriesLogger():
    """Log time series data to CSV file."""

    def __init__(self, filename, label, buffer_size=100):
        self.filename = filename
        self.written_catalog = False
        self.label = label
        self.buffer = []
        self.buffer.append('step,time,{}\n'.format(self.label))
        self.buffer_size = buffer_size
        log.info('Time series log to {}'.format(filename))
        pass

    def add(self, step, value):
        """Add an entry."""
        t = datetime.datetime.now()
        self.buffer.append('{:d},{},{}\n'.format(
            step, t.isoformat(), value))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

        pass

    def flush(self):
        """Write the buffer to file."""

        if not self.written_catalog:
            folder = os.path.dirname(self.filename)
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            catalog = os.path.join(folder, 'catalog')
            basename = os.path.basename(self.filename)
            if not os.path.exists(catalog):
                with open(catalog, 'w') as f:
                    f.write('filename\n');
                    f.write('{}\n'.format(basename))
            else:
                with open(catalog, 'a') as f:
                    f.write('{}\n'.format(basename))
            self.written_catalog = True

        if not os.path.exists(self.filename):
            mode = 'w'
        else:
            mode = 'a'
        with open(self.filename, mode) as f:
            f.write(''.join(self.buffer))
        self.buffer = []

        pass

    def close(self):
        """Flush the rest."""
        self.flush()

        pass
