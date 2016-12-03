import os
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil

def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

        
def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)



def fast_hist(a, b, n):
    hist = confusion_matrix(a, b, np.array([0, 1]))
    return hist

def ComputeHist(blob_gt, blob_pred, perblob = False):
    
    n_cl = len(np.unique(blob_gt))
    if not perblob:
        hist = fast_hist(blob_gt.flatten(),
                          blob_pred.flatten(),
                          n_cl)
    else:
        hist = []
        for i in range(blob_gt.shape[0]):
            hist.append(fast_hist(blob_gt[i,:,:,:].flatten(),
                                  blob_pred[i,:,:,:].flatten(),
                                  n_cl))
    return hist



def CleanTemp(folder):
    if CheckExistants(folder):
        shutil.rmtree(folder)