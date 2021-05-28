import numpy as np


def transform(x, bits):
    i = 0.5 ** (bits - 1)
    return (x + 2 - i) / (4 - 2 * i) * (2 ** bits - 1)


print(transform(np.array([-1.0, 1.0]), 1))
print(transform(np.array([-1.5, -0.5, 0.5, 1.5]), 2))
print(transform(np.array([-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]), 3))
