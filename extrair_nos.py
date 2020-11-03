import numpy as np
from scipy.spatial import KDTree

nos_id = np.array([[0, -2],
                   [252, -2],
                   [504, -2]])

nos_kdtree = KDTree(np.load('MALHA_nos.npy'))

for no in nos_kdtree.query(nos_id)[1]:
    print(no)
