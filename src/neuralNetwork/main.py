from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from pybrain.datasets.classification           import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

# from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

with open('iris.data', 'r') as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = [x.split(',') for x in content]

trainingSet = []

for x in content:
    classification = x[len(x)-1]
    trainingData = [float(x[i]) for i in range(0, len(x) - 1)]
    tup = (trainingData, classification)

    trainingSet.append(tup)

print(len(trainingSet[0][0]))

class_labels = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']

def getIndexOfClassLabel(label):
    for x in range(0, len(class_labels) - 1):
        if (class_labels[x] == label):
            return x
    return -1



# Create data set object
dataSet = ClassificationDataSet( len(trainingSet[0][0]), 1, len(class_labels))

for x in trainingSet:
    dataSet.addSample(x[0], getIndexOfClassLabel(x[1]))

tstdata, trndata = dataSet.splitWithProportion( 0.25 )

# trndata._convertToOneOfMany( )
# tstdata._convertToOneOfMany( )

#Create new FFNN object
fnn = FeedForwardNetwork()

# Construct layers
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(2)

# Add layers to the NN
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

# Connect layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# Add connections the NN
fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)

fnn.sortModules()

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(20000):
    trainer.trainEpochs(1)

    trnresult = percentError( trainer.testOnClassData(), True)
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata), True)

    print("epoch: %4d" % trainer.totalepochs + "  train error: %5.2f%%" % trnresult + "  test error: %5.2f%%" % tstresult)



