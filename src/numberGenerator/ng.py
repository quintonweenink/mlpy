import six
import abc

@six.add_metaclass(abc.ABCMeta)
class NG(object):

    @abc.abstractmethod
    def random(self):
        pass