import numpy as np
#
# def addBias(vec):
#     return np.insert(vec, len(vec[0]), 1.0, axis=1)
#
# # input data
# x = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])
#
# # output data
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])
#
# np.random.seed(1)
#
# syn0 = (2 * np.random.random((3,5))) - 1
# print(syn0)
# syn1 = (2 * np.random.random((5,1))) - 1
# print(syn1)
#
# # trainig step
# for i in range(1):
#     l0 = addBias(x)
#     print(l0)
#     l1 = nonlin(np.dot(l0, syn0))
#     print(l1)
#     l1 = addBias(l1)
#     print(l1)
#     l2 = nonlin(np.dot(addBias(l1), syn1))
#     print(l2)
#
#     l2_error = y - l2
#
#     if(i % 1000) == 0:
#         print("Error:" + str(np.mean(np.abs(l2_error))))
#
#     l2_delta = l2_error * nonlin(l2, deriv=True)
#
#     l1_error = l2_delta.dot(syn1.T)
#
#     l1_delta = l1_error * nonlin(l1, deriv=True)
#
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += l0.T.dot(l1_delta)
#
# print("Output after training")
#
# print(l2)

def nonlin(x, deriv = False):
    if(deriv == True):
        return nonlin(x) * (1 - nonlin(x))

    return 1 / (1 + np.exp(-x))

class FeedForwardNeuralNetwork(object):
    def __init__(self):
        self._layers = []

    def appendLayer(self, layer):
        assert isinstance(layer, Layer)

        self._layers.append(layer)

    def canFire(self):
        return len(self._layers) > 1

    def fire(self, input):
        assert isinstance(input, (np.ndarray, np.generic))
        assert len(input) == self._layers[0].size
        assert self.canFire()

        self._layers[0].result = input

        print(self._layers[0].syn)
        print(self._layers[1].syn)

        self._layers[0].syn = np.array([[0.8, 0.2],
                                        [0.4, 0.9],
                                        [0.3, 0.5]])

        self._layers[1].syn = np.array([[0.3, 0.3, 0.6]])

        print(self._layers[0].syn)
        print(self._layers[1].syn)

        for layer in self._layers:
            layer.propagate()

        return self._layers[len(self._layers) - 1].result

    def backPropagation(self, target):
        for layer in reversed(self._layers):
            target = layer.backPropagate(target)

        return self._layers[len(self._layers) - 1].result

    def toString(self):
        res = ""
        for layer in self._layers:
            res += layer.toString()
        return res

class Layer(object):
    def __init__(self, size, prev, bias = False, label = "Layer"):
        assert isinstance(size, int) and size > 0
        assert isinstance(prev, Layer) or prev == None
        assert isinstance(bias, bool)
        assert isinstance(label, str)

        self.label = label
        self.prev = prev
        self.size = size

        self.result = np.zeros(size)

        self.sum = None
        self.error = None
        self.syn = None
        self.deltaOutputSum = None
        self.deltaWeights = None

        if not prev == None:
            np.random.seed(1)
            prev.syn = (2 * np.random.random((self.size, prev.size))) - 1
            self.prev.deltaOutputSum = np.zeros(self.size)
            self.prev.deltaWeights = (2 * np.random.random((self.size, prev.size))) - 1

    def toString(self):
        return ('=== Layer: {label} ===\n' +
                'Syn: \n' +
                '{syn}\n' +
                'Res: \n' +
                '{res}\n' +
                'Error: \n' +
                '{err}\n' +
                '===========================\n').format(label=self.label,
                                           syn=self.syn, res=self.result, err=self.error)

    def propagate(self):
        if self.prev != None:
            self.sum = np.sum(self.prev.syn, axis=1)
            for neuron in range(self.size):
                self.result[neuron] = nonlin(self.sum[neuron])

    def calculateOutputDelta(self, delta):
        print("+ " + self.label)
        self.error = np.subtract(target, self.result)
        print("Error: " + str(self.error))
        for neuron in range(self.size):
            print(self.sum)
            print(self.error)
            self.prev.deltaOutputSum[neuron] = np.multiply(nonlin(self.sum[neuron], deriv=True), self.error[neuron])

        print("Delta Output Sum: " + str(self.prev.deltaOutputSum))

        for neuron in range(len(self.prev.deltaWeights)):
            for synapse in range(len(self.prev.deltaWeights[0])):
                self.prev.deltaWeights[neuron][synapse] = self.prev.deltaOutputSum[neuron] / self.prev.result[synapse]


        print("Target: " + str(target))
        print("Result: " + str(self.result))
        print("Delta: " + str(self.deltaOutputSum))
        print("DeltaWeights: " + str(self.prev.deltaWeights))

        return self.prev.deltaOutputSum

    def backPropagate(self, delta):
        if not np.any(self.syn):
            return self.calculateOutputDelta(delta)
        else:
            print("+ " + self.label)
            self.prev.deltaOutputSum = (2 * np.random.random((len(self.syn), self.syn[0].size))) - 1
            for neuron in range(len(self.syn)):
                for synapse in range(len(self.syn[0])):
                    self.prev.deltaOutputSum[neuron][synapse] = delta[neuron] / self.syn[neuron][synapse]

            print("Delta Output Sum: " + str(self.prev.deltaOutputSum))

            for neuron in range(self.size):
                print(self.prev.deltaOutputSum)
                print(nonlin(self.result[neuron], deriv=True))
                print(self.result[neuron])
                self.prev.deltaOutputSum[neuron] = nonlin(self.result[neuron], deriv=True) * self.prev.deltaOutputSum[neuron]

            print("Delta Output Sum: " + str(self.prev.deltaOutputSum))

            #self.prev.deltaWeights = self.delta.dot(self.delta, self.prev.result)

            print("+ " + self.label)
            print("Target: " + str(delta))
            print("Result: " + str(self.result))
            print("Delta: " + str(self.delta))
            print("Delta Weights: " + str(self.deltaWeights))
            print("Error: " + str(self.error))
            return self.prev.deltaOutputSum



inputLayer = Layer(size = 2, prev = None, bias = True, label = "Input layer")
hiddenLayer = Layer(size = 3, prev = inputLayer, bias = True, label = "Hidden layer")
outputLayer = Layer(size = 1, prev = hiddenLayer, bias = False, label = "Output layer")

fnn = FeedForwardNeuralNetwork()
fnn.appendLayer(inputLayer)
fnn.appendLayer(hiddenLayer)
fnn.appendLayer(outputLayer)

input = np.array([1,1])

output = fnn.fire(input)
print(fnn.toString())

print("Output: ")
print(output)

target = np.array([0])

print("***********")

output = fnn.backPropagation(target)






