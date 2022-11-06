import numpy as np
import numpy.random as random

# numpys argmax just takes the first result if there are several
# that is problematic and the reason for a new argmax func
def argmax(elements):
    maxElement = np.max(elements)
    bestElements = [i for i in range(0, len(elements)) if elements[i] == maxElement]
    return bestElements[random.randint(0, len(bestElements))]
