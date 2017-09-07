# Get datasets from file
import numpy as np
import random

class DataSetTool(object):
    def __init__(self):
        pass

    def getIrisDataSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(0, 1, 2, 3)))
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=4))
        classifications = []
        for item in outputClassification:
            notInList = True
            for classification in classifications:
                if(classification == item):
                    notInList = False
            if notInList:
                classifications.append(item)
        output = []
        for item in outputClassification:
            target = []
            for classification in classifications:
                if item == classification:
                    target.append(1)
                else:
                    target.append(0)
            output.append(target)
        output = np.array(output)
        max = np.amax(input)
        input = input / max
        input_target = []
        for i in range(len(input)):
            input_target.append((input[i], output[i]))
        random.shuffle(input_target)
        random.shuffle(input_target)
        training = input_target[:int(len(input_target)/2)]
        testing = input_target[int(len(input_target)/2):]

        return training, testing