import numpy as np

random = np.random.random
randint = np.random.randint


# numpys argmax just takes the first result if there are several
# that is problematic and the reason for a new argmax func
def argmax(elements):
    best_elements = argmax_multi(elements)
    return best_elements[randint(0, len(best_elements))]


def argmax_multi(elements):
    max_element = np.max(elements)
    best_elements = [i for i in range(0, len(elements)) if elements[i] == max_element]
    return best_elements
