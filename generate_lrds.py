import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.io import loadmat, savemat

k = 3
xx = 1
LRDS = np.zeros((512, 512, 100))
for n in range(100):
    v = np.zeros((512, 512))
    for jj in range(512//k):
        for ii in range(512//k):
            r = random.sample(range(k**2), k**2)
            mask = np.zeros((k, k))
            mask[np.unravel_index(r[:xx], (k, k))] = 1
            v[3*ii:3*ii+3, 3*jj:3*jj+3] = mask
    LRDS[:, :, n] = v

savemat('LRDS.mat', {'LRDS': LRDS})

cc = LRDS[:, :, 99]
plt.imshow(cc)
plt.show()