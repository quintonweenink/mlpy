# Get datasets from file
import numpy as np
import random

class DataSetTool(object):
    def __init__(self):
        pass

    def getIrisDataSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(0, 1, 2, 3)))
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=4))

        return self.prepairData(input, outputClassification)

    def getGlassDataSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))) # I don't care about the ID
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=10))

        return self.prepairData(input, outputClassification)

    def getPrimaIndiansDiabetesSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7)))
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=8))

        return self.prepairData(input, outputClassification)

    def getHeartDataSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=13))

        return self.prepairData(input, outputClassification)

    def getWineDataSets(self, filePath):
        input = np.array(np.genfromtxt(filePath, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)))
        outputClassification = np.array(np.genfromtxt(filePath, dtype=str, delimiter=',', usecols=0))

        return self.prepairData(input, outputClassification)

    def prepairData(self, input, outputClassification):
        output, classifications = self.associateClassifications(input, outputClassification)
        input = self.scaleInput(input)

        input_target = self.associateInputAndOutput(input, output)

        random.shuffle(input_target)
        random.shuffle(input_target)

        return self.devideUpData(input_target)

    def associateClassifications(self, input, outputClassification):
        classifications = []
        for item in outputClassification:
            notInList = True
            for classification in classifications:
                if (classification == item):
                    notInList = False
            if notInList:
                classifications.append(item)
        output = []
        for item in outputClassification:
            target = []
            for classification in classifications:
                if item == classification:
                    target.append(0.9)
                else:
                    target.append(0.1)
            output.append(target)
        output = np.array(output)

        return output, classifications

    def associateInputAndOutput(self, input, output):
        input_target = []
        for i in range(len(input)):
            input_target.append((input[i], output[i]))
        return input_target

    def devideUpData(self, data):
        size = int(len(data) / 3)

        training = data[:size*2]
        testing = None
        generalization = data[size*2:]

        return training, testing, generalization

    def scaleInput(self, input):
        max = np.amax(input, axis=0)
        min = np.amin(input, axis=0)
        return (((input - min) / (max - min)) * 2) - 1