import log_manager
import logger

log = logger.get()

class LazyRegisterer():

    def __init__(self, fname, typ, name):
        self._fname = fname
        self._typ = typ
        self._name = name
        self._registered = False
        log.info('{} type data "{}" log to "{}"'.format(typ, name, fname))

        pass

    def is_registered(self):

        return self._registered

    def get_fname(self):

        return self._fname

    def register(self):
        self._registered = True
        log_manager.register(self._fname, self._typ, self._name)

        pass
