from ng import NG

class CPRNG(NG):
    __hair = 'Fluffy'

    def __init__(self):
        self.__hair = hair

    def toString(self):
        return self.getName() + ' has ' + self.__hair + ' hair'