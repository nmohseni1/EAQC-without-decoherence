import sys
print('python: ' + sys.version)
print('')

import numpy
import qutip
import scipy

modules = [numpy, qutip, scipy]
for m in modules :
    print(m.__name__ + ': ' + m.__version__)
