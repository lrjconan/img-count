import datetime
import log_manager
import logger
import os

log = logger.get()


class TimeSeriesLogger():
    """Log time series data to CSV file."""

    def __init__(self, filename, label, name=None, buffer_size=100):
        self.filename = filename
        self.written_catalog = False
        if name is None:
            self.name = label
        else:
            self.name = name
        self.label = label
        self.buffer = []
        self.buffer.append('step,time,{}\n'.format(self.label))
        self.buffer_size = buffer_size
        log.info('Time series data "{}" log to "{}"'.format(label, filename))
        pass

    def add(self, step, value):
        """Add an entry."""
        t = datetime.datetime.utcnow()
        self.buffer.append('{:d},{},{}\n'.format(
            step, t.isoformat(), value))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

        pass

    def flush(self):
        """Write the buffer to file."""

        if not self.written_catalog:
            log_manager.register(self.filename, 'csv', self.name)
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
