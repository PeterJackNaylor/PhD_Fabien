
from scipy.ndimage.morphology import morphological_gradient
from skimage.morphology import reconstruction
import numpy as np
from WrittingTiff.createfold import PredImageFromNet
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from Deprocessing.Morphology import PostProcess
from math import ceil
import pdb
from skimage.morphology import dilation, disk
from scipy.misc import imsave

def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad


def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for y in xrange(0, image.shape[0] - windowSize[0] + stepSize, stepSize):
        for x in xrange(0, image.shape[1] - windowSize[1] + stepSize, stepSize):
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

def RemoveBordersByReconstruction(Img, BorderSize = 1, left=False, right=False, top=False, bottom=False):
    g = Img.copy()
    g[BorderSize:-BorderSize, BorderSize:-BorderSize] = 0
    if left:
        g[:, 0] = 0
    if right:
        g[:,-1] = 0
    if top:
        g[0,:] = 0
    if bottom:
        g[-1,:] = 0
    ToRemove = reconstruction(g, Img, 'dilation')
    return Img - ToRemove, np.mean(ToRemove)

def PJ_clear_border(bin_img, left, right, top, bottom):
    bin = bin_img.copy()
    if left:
        left_side = bin[:, 0].copy()
        bin[:, 0] = 0
    if right:
        right_side = bin[:, -1].copy()
        bin_img[:, -1] = 0
    if top:
        top_side = bin[0, :].copy()
        bin_img[0, :] = 0
    if bottom:
        bottom_side = bin[-1, :].copy()
        bin_img[-1, :] = 0

    bin = clear_border(bin, bgval = 0)

    if left:
        bin[:, 0] = left_side
    if right:
        bin[:, -1] = right_side
    if top:
        bin[0, :] = top_side
    if bottom:
        bin[-1, :] = bottom_side

    return bin

def PredLargeImageFromNet(net_1, image, stepSize, windowSize, removeFromBorder=10, 
                          method="avg", param=7, ClearBorder="RemoveBorderObjects",
                          threshold = 0.5):

    #pdb.set_trace() 
    x_s, y_s, z_s = image.shape
    dim_result = 2
    
    if method == "median":
        dim_result = int((ceil(float(windowSize[0]) / stepSize) + 1) * (ceil(float(windowSize[1]) / stepSize) + 1)) + 3
        counter = np.zeros(shape=(x_s, y_s))
        
    result = np.zeros(shape=(x_s, y_s, dim_result))
    if method == "median":
        result -= 1 
    thresh_list = []

    for x_b, y_b, x_e, y_e, window in sliding_window(image, stepSize, windowSize):
        left, right, bottom, top = False, False, False, False
        if x_b == 0:
            left = True
        if x_e == x_s:
            right = True
        if y_b == 0:
            top = True
        if y_e == y_s:
            bottom = True


        prob_image1, bin_image1 = PredImageFromNet(net_1, window, with_depross=True)
        val = removeFromBorder if ClearBorder == "Reconstruction" else 0
        
        y_b += val
        y_e -= val
        x_b += val
        x_e -= val

        inter_result = prob_image1[val:-val, val:-val] if val!= 0 else prob_image1

        if ClearBorder == "RemoveBorderObjects":

            inter_bin = (inter_result > 0.5 + 0.0).astype(np.uint8)
            inter_bin_temp = PJ_clear_border(inter_bin, left, right, top, bottom)
            inter_bin[inter_bin > 0] = 1
            inter_bin_temp[inter_bin_temp > 0] = 1
            removed_cells = inter_bin - inter_bin_temp
            removed_cells = dilation(removed_cells, disk(2))
            inter_result[removed_cells == 1] = 0
            inter_bin = 1 - removed_cells.copy()

        elif ClearBorder == "RemoveBorderWithDWS":

            inter_bin = PostProcess(inter_result, param)
            inter_bin_without = PJ_clear_border(inter_bin, left, right, top, bottom)
            removed_cells = inter_bin - inter_bin_without
            removed_cells[removed_cells > 0] = 1  
            removed_cells = dilation(removed_cells, disk(2))
            inter_result[removed_cells == 1] = 0
            inter_bin = 1 - removed_cells.copy()

        elif ClearBorder == "Reconstruction":
            
            inter_result, thresh = RemoveBordersByReconstruction(inter_result, removeFromBorder, left, right, top, bottom)
            thresh_list += [thresh]
            if method == "avg" or method == "median":
                print "avg nor median are implemented with Reconstruction for clear border, switched to max"
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
                if el != dim_result:
                    y, x = np.where(counter[y_b:y_e, x_b:x_e] == el)
                    x_img = x + x_b
                    y_img = y + y_b
                    result[y_img, x_img, int(el)] = inter_result[y, x] # should be inversed
            counter[y_b:y_e, x_b:x_e] += 1

    if method == "avg" :
        x, y = np.where(result[:,:,1] == 0)
        result[x, y, 1] = 1
        prob_map = result[:, :, 0] / result[:, :, 1]
        
    elif method == "max":
        prob_map = result[:, :, 0].copy()

    elif method == "median":
        prob_map = np.zeros(shape=(x_s, y_s), dtype='float64')
        done_map = np.zeros(shape=(x_s, y_s))
        for el in range(dim_result - 1 , -1, -1):
            x, y = np.where(result[:,:, int(el)] != -1 )
            tmp_done_map = np.zeros(shape=(x_s, y_s))
            tmp_done_map[x, y] = 1
            diff_done_map = tmp_done_map.copy() - done_map.copy()
            x_add, y_add = np.where(diff_done_map > 0)
            done_map[x_add, y_add] = 1
            prob_map[x_add, y_add] = np.median(result[x_add, y_add, 0:(int(el)+1)], axis=1)
	    #imsave("diff_map_{}.png".format(el), diff_done_map)
    if ClearBorder == "Reconstruction":

        threshold = threshold - np.mean(thresh_list)

    bin_map = prob_map > threshold + 0.0
    bin_map = bin_map.astype(np.uint8)
    return prob_map, bin_map, threshold

