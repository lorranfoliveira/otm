import numpy as np
from scipy.spatial import KDTree

y = np.array(
    [2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232,
     242, 252, 262, 272, 282, 292, 302, 312, 322, 332, 342, 352, 362, 372])
x = np.full(len(y), 2)

nos = np.array([x, y]).T

nos = np.array([[507, -5],
                [-3, 253]])

nos_kdtree = KDTree(np.load('MALHA_nos.npy'))

print(list(nos_kdtree.query(nos)[1]))
