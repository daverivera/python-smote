# Author: Dave Rivera <daverivera90@gmail.com>
#         Julian David Arias Londoño  <jdarias@udea.edu.co>

import numpy as np
from SMOTE import SMOTE
import matplotlib.pyplot as plt

n_rows = 30

T = np.random.randn(n_rows,2)

c = np.ones((n_rows,1)) # Class
T = np.append(T, c, axis=1)

N = 10
k = 5
smote = SMOTE(T,N,k=k)
synth = smote.over_sampling()
print('# Synth Samps: ', synth.shape[0])


plt.title('SMOTE')
plt.xlabel('Attr 1')
plt.ylabel('Attr 2')
plt.scatter(T[:, 0], T[:, 1], marker='x')
plt.scatter(synth[:, 0], synth[:, 1], marker='x', color='red')
plt.show()
