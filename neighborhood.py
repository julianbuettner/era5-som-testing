#!/usr/bin/env python3

import matplotlib.pyplot as plt
from math import e


def f(x):
    return e ** -(x*x/2)


fig, ax = plt.subplots()
x = [xx / 100 for xx in range(0, 200)]
y = [f(xx) for xx in x]
plt.plot(x, y)
plt.ylabel('f(x)')
plt.xlabel('x')
plt.grid()
ax.set_ylim([0, 1])
plt.show()
