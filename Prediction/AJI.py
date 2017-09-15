from skimage.measure import label
from skimage.io import imread
import numpy as np
import pdb
import time
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
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
    indice = np.array(res).argmax() + 1

    return indice

def AJI(G, S):
    """
    AJI as described in the paper, AJI is more abstract implementation but 100times faster
    """
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
        USED[j - 1] = 1

    def h(indice):
        if USED[indice - 1] == 1:
            return 0
        else:
            only_prediction = np.zeros_like(S)
            only_prediction[ S == indice ] = 1
        return only_prediction.sum()
    U_sum = map(h, range(1, S.max() + 1))
    U += np.sum(U_sum)
    return float(C) / float(U)  



pbar2 = ProgressBar()
def AJI_fast(G, S):
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0 
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()
        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union
        
        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)  





# USED = 1 for used and 0 for unused
if __name__ == '__main__':
    PATH = "/Users/naylorpeter/Desktop/Results/Slide___Breast_1_*-_classic"
    lb = PATH + "/Label.png"
    p = PATH + "/Bin.png"
    GT = imread(lb)
    pred = imread(p)
    #pdb.set_trace()
    start_time = time.time()

    #score = AJI(GT, pred)

    diff_time = time.time() - start_time

    print ' \n '
    #print 'AJI: {}'.format(score)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    start_time = time.time()

    score = AJI_fast(GT, pred)

    diff_time = time.time() - start_time

    print ' \n '
    print 'AJI_fast:{}'.format(score)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
