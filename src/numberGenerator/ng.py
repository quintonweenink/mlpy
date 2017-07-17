import six
import abc

@six.add_metaclass(abc.ABCMeta)
class NG:

    @abc.abstractmethod
    def getRandomSet(self):
        pass

    @abc.abstractmethod
    def toString(self):
        pass

    @abc.abstractmethod
    def random(self):
        pass