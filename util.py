import numpy as np
import numpy.random as random


# numpys argmax just takes the first result if there are several
# that is problematic and the reason for a new argmax func
def argmax(elements):
    max_element = np.max(elements)
    best_elements = [i for i in range(0, len(elements)) if elements[i] == max_element]
    return best_elements[random.randint(0, len(best_elements))]
