from skimage.measure import label
import numpy as np
import pdb
import time
from progressbar import ProgressBar
pbar = ProgressBar()


def Intersection(A, B):
    C = A + B
    C[C != 2] = 0
    C[C == 2] = 1
    return C

def Union(A, B):
    C = A + B
    C[C > 0] = 1
    return C


def AssociatedCell(G_i, S):
    def g(indice):
        S_indice = np.zeros_like(S)
        S_indice[ S == indice ] = 1
        NUM = float(Intersection(G_i, S_indice).sum())
        DEN = float(Union(G_i, S_indice).sum())
        return NUM / DEN
    res = map(g, range(1, S.max() + 1))

    return np.array(res).argmax() + 1

def AJI(G, S):
    G = label(G, background=0)
    S = label(S, background=0)

    C = 0
    U = 0 
    USED = np.zeros(S.max())

    for i in pbar(range(1, G.max() + 1)):

        only_ground_truth = np.zeros_like(G)
        only_ground_truth[ G == i ] = 1

        j = AssociatedCell(only_ground_truth, S)
        only_prediction = np.zeros_like(S)
        only_prediction[ S == j ] = 1

        C += Intersection(only_prediction, only_ground_truth).sum()
        U += Union(only_prediction, only_ground_truth).sum()
        USED[j] = 1

    def h(indice):
        if USED[indice - 1] == 1:
            return 0
        else:
            only_prediction = np.zeros_like(S)
            only_prediction[ S == indice ] = 1
            return only_prediction.sum()
    U_sum = map(h, range(1, S.max() + 1))
    U += np.sum(U_sum)
    pdb.set_trace()
    return float(C) / float(U)  



# USED = 1 for used and 0 for unused
