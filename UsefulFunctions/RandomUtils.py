import os
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil
from skimage.morphology import erosion, disk
def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

        
def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)



def fast_hist(a, b, n):
    hist = confusion_matrix(a, b, np.array(list(range(n))))
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

def textparser(file):
    with open(file) as f:
        content = f.readlines()
    res = {}
    for el in content:
        key, val = el.split(':')
        val = val[3:-4]
        res[key] = float(val)
    return res
    
def color_bin(bin_labl):
    dim = bin_labl.shape
    x, y = dim[0], dim[1]
    res = np.zeros(shape=(x, y, 3))
    for i in range(1, bin_labl.max() + 1):
        rgb = np.random.normal(loc = 125, scale=100, size=3)
        rgb[rgb < 0 ] = 0
        rgb[rgb > 255] = 255
        rgb = rgb.astype(np.uint8)
        res[bin_labl == i] = rgb
    return res.astype(np.uint8)

def add_contours(rgb_image, contour, ds = 2):
    """
    The image has to be a binary image 
    """
    rgb = rgb_image.copy()
    contour[contour > 0] = 1
    boundery = contour - erosion(contour, disk(ds))
    rgb[boundery > 0] = np.array([0, 0, 0])
    return rgb

