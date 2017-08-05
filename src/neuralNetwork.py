from pybrain.pybrain.structure.networks.feedforward import FeedForwardNetwork

n = FeedForwardNetwork()

from pybrain.pybrain.structure.modules import LinearLayer, SigmoidLayer

# Create Layers
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

# Add layers to the Network
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

from pybrain.pybrain.structure.connections import FullConnection

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)


n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()

print(n)

n.activate([1, 2])

#DS = ClassificationDataSet(2, class_labels =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])