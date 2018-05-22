import numpy as np
import random


def random_add(n):
    return n + random.randint(1, 100)


def random_add_array(arr):
    return arr + np.random.uniform(0.0, 1.0, size=arr.size)
