import numpy as np

def nonlin(x, deriv = False):
    if(deriv == True):
        return nonlin(x) * (1 - nonlin(x))

    return 1 / (1 + np.exp(-x))

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
                'Delta Weights: \n' +
                '{delta}\n' +
                '===========================\n').format(label=self.label,
                                           syn=self.syn, res=self.result, err=self.error, delta=self.deltaWeights)

    def propagate(self):
        if self.prev != None:
            self.sum = np.sum(self.prev.syn, axis=1)
            for neuron in range(self.size):
                self.result[neuron] = nonlin(self.sum[neuron])

    def calculateOutputDelta(self, delta):
        self.error = np.subtract(delta, self.result)
        for neuron in range(self.size):
            self.prev.deltaOutputSum[neuron] = np.multiply(nonlin(self.sum[neuron], deriv=True), self.error[neuron])

        for neuron in range(len(self.prev.deltaWeights)):
            for synapse in range(len(self.prev.deltaWeights[0])):
                self.prev.deltaWeights[neuron][synapse] = self.prev.deltaOutputSum[neuron] / self.prev.result[synapse]

        return self.prev.deltaOutputSum

    def backPropagate(self, delta):
        if not np.any(self.syn):
            return self.calculateOutputDelta(delta)
        elif not self.prev == None:
            # Construct arrays
            self.prev.deltaOutputSum = (2 * np.random.random((len(self.syn), self.syn[0].size))) - 1
            self.prev.deltaWeights = (2 * np.random.random((len(self.prev.syn), self.prev.syn[0].size))) - 1

            for neuron in range(len(self.syn)):
                for synapse in range(len(self.syn[0])):
                    self.prev.deltaOutputSum[neuron][synapse] = delta[neuron] / self.syn[neuron][synapse]

            for neuron in range(len(self.syn)):
                self.prev.deltaOutputSum[neuron] = np.multiply(nonlin(self.result[neuron], deriv=True), self.prev.deltaOutputSum[neuron])


            for synapse in range(self.size):
                for neuron in range(self.prev.size):
                    for item in range(len(self.prev.deltaOutputSum)):
                        self.prev.deltaWeights[synapse][item] = self.prev.deltaOutputSum[item][synapse] / self.prev.result[neuron]

            return self.prev.deltaOutputSum

    def applyDeltaWeights(self):
        if np.any(self.syn):
            for neuron in range(len(self.syn)):
                for synapse in range(len(self.syn[0])):
                    self.syn[neuron][synapse] += self.deltaWeights[neuron][synapse]