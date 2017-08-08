# -*- coding: utf-8 -*-

import time
import glob
import os
from scipy.misc import imread, imsave
import numpy as np
from createfold import GetNet, PredImageFromNet, DynamicWatershedAlias, dilation, disk, erosion
import caffe
import pdb
from Deprocessing.Transfer import ChangeEnv
import progressbar
from scipy.ndimage.morphology import morphological_gradient
from skimage.segmentation import clear_border
from skimage.morphology import reconstruction, remove_small_objects
from UsefulFunctions.UsefulImageConstruction import PredLargeImageFromNet, RemoveBordersByReconstruction

stepSize = 10
windowSize = (224 , 224)
param = 5




def pred_f(image, net1, net2, stepSize=stepSize, windowSize=windowSize,
           param=param, border=1, method="avg", borderImage = "Reconstruction",
           return_all = False):
    prob_image1, bin_image1, threshold1 = PredLargeImageFromNet(net1, image, stepSize, windowSize,
                                                                removeFromBorder=border,
                                                                method=method,
                                                                param=param,
                                                                ClearBorder= borderImage)


    if net2 is not None:

        prob_image2, bin_image2, threshold2 = PredLargeImageFromNet(net2, image, stepSize, windowSize,
                                                                    removeFromBorder=border,
                                                                    method=method,
                                                                    param=param,
                                                                    ClearBorder= borderImage)
        thresh = ( threshold1 + threshold2 ) / 2.
        prob = ( prob_image1 + prob_image2 ) / 2.
    else:
        thresh = threshold1 
        prob = prob_image1 
    if not return_all:
        return prob, thresh
    else:
        return {"model1":(prob_image1, threshold1), "model2":(prob_image2, threshold2),
                "ensemble":(prob, thresh)}


def PredOneImage(path, outfile, c, f, net1, net2, ClearSmallObjects=None):
    # pdb.set_trace()
    if not os.path.isfile(outfile):
        image = imread(path)[:,:,0:3]
        image = c(image)
        prob, thresh = f(image, net1, net2, method="max")
        imsave(outfile.replace('.png','_raw.png'), image)
        imsave(outfile.replace('.png', '_prob.png'), prob)
        for param in range(6,10,2):
            image_seg = PostProcess(prob,param)
            if ClearSmallObjects is not None:
               image_seg = remove_small_objects(image_seg, ClearSmallObjects)
            ContourSegmentation = Contours(image_seg)
            x_, y_ = np.where(ContourSegmentation > 0)
            image_segmented = image.copy()
            image_segmented[x_,y_,:] = np.array([0,0,0])
            imsave(outfile.replace('.png', '_wsl_{}.png').format(param), image_segmented)
    else:
        #print "Files exists"
        pass

def crop(image):
    return image[50:-30,:]

    
if __name__ == "__main__":
    
    PATH = "/data/users/pnaylor/Documents/Python/Francois/Only_one_images/*40x.png"
    OUT = "/data/users/pnaylor/Documents/Python/Francois/ClearBorders"

    ClearSmallObjects = 50

    ImagesToProcess = glob.glob(PATH)
    caffe.set_mode_gpu()
    cn_1 = "FCN_0.01_0.99_0.005"
    wd_1 = "/data/users/pnaylor/Documents/Python/Francois" #"/share/data40T_v2/Peter/pretrained_models"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    wd_2 = wd_1
    ChangeEnv("/data/users/pnaylor/Bureau/ToAnnotate", os.path.join(wd_1, cn_1))
    ChangeEnv("/data/users/pnaylor/Bureau/ToAnnotate", os.path.join(wd_1, cn_2))
    net_1 = GetNet(cn_1, wd_1)
    net_2 = GetNet(cn_2, wd_1)
    #net_2 = None
    progress = progressbar.ProgressBar()
    TIME = time.time()
    for path in progress(ImagesToProcess):
        outfile = os.path.basename(path)
        outfile = os.path.join(OUT, outfile)
        PredOneImage(path, outfile, crop,  pred_f, net_1, net_2, ClearSmallObjects)
    diff_time = time.time() - TIME
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)



