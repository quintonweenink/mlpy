import matplotlib.pyplot as plt
import neuralNetwork.structure.activations as activations


y = -6
arry = []
arrx = []

while y <= 6:
    arry.append(y)
    arrx.append(activations.nonlin(y))
    y += 0.01


plt.grid(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.ion()

plt.plot(arry, arrx)

plt.show(5)