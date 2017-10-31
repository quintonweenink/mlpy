import matplotlib.pyplot as plt
import neuralNetwork.structure.activations as activations


y = -5
arry = []
arrx = []

print(activations.nonlin(-5))

while y <= 5:
    arry.append(y)
    arrx.append(activations.nonlin(y))
    y += 0.01


plt.grid(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.ion()

plt.plot(arry, arrx)

plt.show(5)