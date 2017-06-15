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
from skimage.morphology import remove_small_objects


stepSize = 10
windowSize = (224 , 224)
param = 5

def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad


def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[1]:
                y = image.shape[0] - windowSize[1]
                change = True
            if res_img.shape[1] != windowSize[0]:
                x = image.shape[1] - windowSize[0]
                change = True
            if change:
                res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)

def RemoveBordersByReconstruction(Img, BorderSize = 1):
    g = Img.copy()
    g[BorderSize:-BorderSize, BorderSize:-BorderSize] = 0
    ToRemove = reconstruction(g, Img, 'dilation')
    return Img - ToRemove, np.mean(ToRemove)


def PredLargeImageFromNet(net_1, image, stepSize, windowSize, removeFromBorder = 10, method="avg", ClearBorder = "RemoveBorderObjects", threshold = 0.5):
    #pdb.set_trace() 
    x_s, y_s, z_s = image.shape
    result = np.zeros(shape=(x_s, y_s, 2))
    thresh_list = []
    for x_b, y_b, x_e, y_e, window in sliding_window(image, stepSize, windowSize):
        prob_image1, bin_image1 = PredImageFromNet(net_1, window, with_depross=True)
        val = removeFromBorder
        y_b += val
        y_e -= val
        x_b += val
        x_e -= val


        inter_result = prob_image1[val:-val, val:-val] if val!= 0 else prob_image1

        if ClearBorder == "RemoveBorderObjects":
            inter_bin = (inter_result > 0.5 + 0.0).astype(np.uint8)
            inter_bin = clear_border(inter_bin)
            inter_result[inter_bin == 0] = 0
        elif ClearBorder == "RemoveBorderWithDWS":
        # pdb.set_trace()
            inter_bin = PostProcess(inter_result, param)
            inter_bin_without = clear_border(inter_bin, bgval = 0)
            inter_bin_without = inter_bin - inter_bin_without
            inter_bin_without[inter_bin_without > 0] = 1  
            inter_result[inter_bin_without == 1] = 0
            inter_bin = 1 - inter_bin_without.copy()
        elif ClearBorder == "Reconstruction":
            
            inter_result, thresh = RemoveBordersByReconstruction(inter_result, removeFromBorder)
            thresh_list += [thresh]
            if method == "avg":
                print "avg not implemented with Reconstruction for clear border, switched to max"
                method = "max"

        if method == 'avg':
            inter_mean = np.ones_like(inter_result)
            inter_result[inter_bin == 0] = 0
            inter_mean[ inter_bin == 0 ] = 0
            result[y_b:y_e, x_b:x_e, 0] += inter_result
            result[y_b:y_e, x_b:x_e, 1] += inter_mean

        elif method == "max":
            result[y_b:y_e, x_b:x_e, 1] = inter_result
            result[y_b:y_e, x_b:x_e, 0] = np.max(result[y_b:y_e, x_b:x_e,:], axis=2)

    if method == "avg" :
        x, y = np.where(result[:,:,1] == 0)
        result[x, y, 1] = 1
        prob_map = result[:, :, 0] / result[:, :, 1]
        
    elif method == "max":
        prob_map = result[:, :, 0].copy()

    if ClearBorder == "Reconstruction":

        threshold = threshold - np.mean(thresh_list)

    bin_map = prob_map > threshold + 0.0
    bin_map = bin_map.astype(np.uint8)
    return prob_map, bin_map, threshold


def pred_f(image, net1, net2, stepSize=stepSize, windowSize=windowSize,
           param=param, border=1, method="avg", borderImage = "Reconstruction"):
    prob_image1, bin_image1 , threshold1 = PredLargeImageFromNet(net1, image, stepSize, windowSize, border, method, borderImage)
    prob_image2, bin_image1 , threshold2 = PredLargeImageFromNet(net1, image, stepSize, windowSize, border, method, borderImage)
    

    thresh = ( threshold1 + threshold2 ) / 2.
    prob = ( prob_image1 + prob_image2 ) / 2.
    return prob, thresh

def PostProcess(prob_image, param=param, thresh = 0.5):
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh)
    return segmentation_mask


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



