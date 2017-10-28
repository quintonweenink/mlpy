import numpy as np

arr = np.array([
[1, 2, 23, 4, 5, 9],
[1, 2, 3, 4, 5, 9],
[1, 34, 3, 4, 5, 9],
[1, 2, 3, 4, 70, 1]
])

arr2 = np.array([
[1, 22, 23, 4, 5, 9],
[1, 2, 3, 4, 223, 9],
[1, 34, 34, 4, 5, 23],
[45, 2, 3, 4, 70, 9]
])

result = arr - arr2

print(np.mean(np.square(result)))

