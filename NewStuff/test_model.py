
import tensorflow as tf 
from DataGenClass import DataGenMulti
from UsefulFunctions.ImageTransf import ListTransform
from scipy.misc import imsave
import os
import numpy as np
import pdb
from UsefulFunctions.UsefulImageConstructionTF import PredLargeImageFromNet, Contours
from UNetBatchNorm_v2 import UNetBatchNorm
from UNetObject_v2 import UNet
from sklearn.metrics import confusion_matrix
from Deprocessing.Morphology import PostProcess
from AJI import AJI


CUDA_NODE = 0
HEIGHT = 212 
WIDTH = 212
CROP = 4
PATH = '/Users/naylorpeter/Documents/Histopathologie/NeerajKumar/ForDatagen'
BATCH_SIZE = 1
S = True



os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

# define here the data

transform_list, transform_list_test = ListTransform()
DG = DataGenMulti(PATH, split='test', crop = 1, size=(1000, 1000),
                   transforms=transform_list_test, UNet=True, num="Breast")

 
key = DG.RandomKey(False)
model = UNetBatchNorm("f",
                          BATCH_SIZE=1, 
                          IMAGE_SIZE = (212,212),
                          NUM_CHANNELS=3, 
                          NUM_LABELS=2,
                          N_FEATURES=32)
META = "/Users/naylorpeter/Desktop/Experiences/UNet/32_0.00005_0.0001/model.ckpt-2500.meta"
stepSize = 212
windowSize = (212 + 184, 212 + 184)
MEAN_FILE = "mean_file.npy"

JACIND = 0

TP, TN, FN, FP = 0, 0, 0, 0

for _ in range(1): #range(DG.length):
    key = DG.NextKeyRandList(key)
    img, anno = DG[key]
    prob_map, bin_map, threshold = PredLargeImageFromNet(model, META, img, stepSize, windowSize, removeFromBorder=0, 
                                                         method="max", param=10, ClearBorder="Classic",
                                                         threshold = 0.5, UNet=True, MEAN_FILE=MEAN_FILE)  
    prob_map = 1 - prob_map
    bin_map[ bin_map > 0 ] = 1
    bin_map[ bin_map == 0 ] = 255
    bin_map[ bin_map == 1 ] = 0
    cm = confusion_matrix(anno.flatten(), bin_map.flatten(), labels=[0, 255]).astype(np.float)
    PP = PostProcess(prob_map, param=10)
    pdb.set_trace()
    JACIND += AJI(anno, PP)
    CellCont = Contours(PP)
    contour_rgb = img[92:-92, 92:-92].copy()
    contour_rgb[CellCont > 0] = np.array([0, 0, 0])
#    imsave("/tmp/{}_annotation.png".format(_), anno)
#    imsave("/tmp/{}_prediction.png".format(_), bin_map)
#    imsave("/tmp/{}_probability.png".format(_), prob_map)
#    imsave("/tmp/{}_contours.png".format(_), contour_rgb)
    TP += cm[1, 1]
    TN += cm[0, 0]
    FN += cm[0, 1]
    FP += cm[1, 0]

precision = float(TP) / (TP + FP) 
print "Precision : {}".format(precision)
recall = float(TP) / (TP + FN)
print "Recall : {}".format(recall)
F1 = 2 * precision * recall / (precision + recall)
print "F1 : {}".format(F1)
acc = float(TP + TN) / (TP + TN + FP + FN)
print "Acc : {}".format(acc)
print "AJI : {}".format(AJI / DG.length)
pdb.set_trace()