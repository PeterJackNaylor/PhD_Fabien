
from scipy.ndimage.morphology import morphological_gradient
from skimage.morphology import reconstruction
import numpy as np
from WrittingTiff.createfold import PredImageFromNet
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from Deprocessing.Morphology import PostProcess
from math import ceil




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


def PredLargeImageFromNet(net_1, image, stepSize, windowSize, removeFromBorder=10, 
                          method="avg", param=7, ClearBorder="RemoveBorderObjects",
                          threshold = 0.5):
    #pdb.set_trace() 
    x_s, y_s, z_s = image.shape
    dim_result = 2
    
    if method == "median":
        dim_result = ceil(float(windowSize[0]) / stepSize) * ceil(float(windowSize[1]) / stepSize) 
        counter = np.zeros(shape=(x_s, y_s))

    result = np.zeros(shape=(x_s, y_s, dim_result))
    if method == "median":
        result -= -1 
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

            inter_bin = PostProcess(inter_result, param)
            inter_bin_without = clear_border(inter_bin, bgval = 0)
            inter_bin_without = inter_bin - inter_bin_without
            inter_bin_without[inter_bin_without > 0] = 1  
            inter_result[inter_bin_without == 1] = 0
            inter_bin = 1 - inter_bin_without.copy()

        elif ClearBorder == "Reconstruction":
            
            inter_result, thresh = RemoveBordersByReconstruction(inter_result, removeFromBorder)
            thresh_list += [thresh]
            if method == "avg" or method == "median":
                print "avg not implemented with Reconstruction for clear border, switched to max"
                method = "max"

        elif ClearBorder == "Classic":
            inter_bin = np.ones_like(inter_result)

        if method == 'avg':

            inter_mean = np.ones_like(inter_result)
            inter_result[inter_bin == 0] = 0
            inter_mean[ inter_bin == 0 ] = 0
            result[y_b:y_e, x_b:x_e, 0] += inter_result
            result[y_b:y_e, x_b:x_e, 1] += inter_mean

        elif method == "max":

            result[y_b:y_e, x_b:x_e, 1] = inter_result
            result[y_b:y_e, x_b:x_e, 0] = np.max(result[y_b:y_e, x_b:x_e,:], axis=2)

        elif method == "median":
            #RAM intensive a priori
            for el in np.unique(counter):
                y, x = np.where(counter[y_b:y_e, x_b:x_e] == el)
                x_img = x + x_b
                y_img = y + y_b
                result[x_img, y_img, el] = inter_result[x, y]
            counter[y_b:y_e, x_b:x_e] += 1

    if method == "avg" :
        x, y = np.where(result[:,:,1] == 0)
        result[x, y, 1] = 1
        prob_map = result[:, :, 0] / result[:, :, 1]
        
    elif method == "max":
        prob_map = result[:, :, 0].copy()

    elif method == "median":
        x, y, z = np.where(result != -1 )
        prob_map = np.median(result[x, y, z], axis=2)

    if ClearBorder == "Reconstruction":

        threshold = threshold - np.mean(thresh_list)

    bin_map = prob_map > threshold + 0.0
    bin_map = bin_map.astype(np.uint8)
    return prob_map, bin_map, threshold

