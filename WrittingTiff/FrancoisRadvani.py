import glob
import os
from scipy.misc import imread, imsave
import numpy as np
from createfold import GetNet, PredImageFromNet, DynamicWatershedAlias, dilation, disk, erosion
import caffe
import pdb
from Deprocessing.Transfer import ChangeEnv
import progressbar





stepSize = 180
windowSize = (224 , 224)
param = 7

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


def PredLargeImageFromNet(net_1, image, stepSize, windowSize):
    #pdb.set_trace() 
    x_s, y_s, z_s = image.shape
    result = np.zeros(shape=(x_s, y_s, 2))
    for x_b, y_b, x_e, y_e, window in sliding_window(image, stepSize, windowSize):
        prob_image1, bin_image1 = PredImageFromNet(net_1, window, with_depross=True)
        result[y_b:y_e, x_b:x_e, 0] += prob_image1
        result[y_b:y_e, x_b:x_e, 1] += 1.
    prob_map = result[:, :, 0] / result[:, :, 1]
    bin_map = prob_map > 0.5 + 0.0
    bin_map = bin_map.astype(np.uint8)
    return prob_map, bin_map

def pred_f(image, net1, net2, stepSize=stepSize, windowSize=windowSize, param=param):
    prob_image1, bin_image1 = PredLargeImageFromNet(net_1, image, stepSize, windowSize)
    prob_image2, bin_image2 = PredLargeImageFromNet(net_2, image, stepSize, windowSize)
    
    #pdb.set_trace()
    segmentation_mask = DynamicWatershedAlias(prob_image1, param)
    segmentation_mask[segmentation_mask > 0] = 1
    contours = dilation(segmentation_mask, disk(2)) - \
        erosion(segmentation_mask, disk(2))

    x, y = np.where(contours == 1)
    image[x, y] = np.array([0, 0, 0])


    return image

def PredOneImage(path, outfile, c, f, net1, net2):
    # pdb.set_trace()
    if not os.path.isfile(outfile):
        #pdb.set_trace()
        image = imread(path)[:,:,0:3]
        image = c(image)
        image = f(image, net1, net2)
        imsave(outfile, image)
    else:
        #print "Files exists"
	pass

def crop(image):
    return image[50:-30,:]

    
if __name__ == "__main__":
    
    PATH = "/data/users/pnaylor/Documents/Python/Francois/New_images_TMA_ICA/*40x.png"
    OUT = "/data/users/pnaylor/Documents/Python/Francois/Out"



    ImagesToProcess = glob.glob(PATH)
    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.005"
    wd_1 = "/data/users/pnaylor/Documents/Python/Francois"#"/share/data40T_v2/Peter/pretrained_models"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    wd_2 = wd_1
    ChangeEnv("/data/users/pnaylor/Bureau/ToAnnotate", os.path.join(wd_1, cn_1))
    ChangeEnv("/data/users/pnaylor/Bureau/ToAnnotate", os.path.join(wd_1, cn_2))
    net_1 = GetNet(cn_1, wd_1)
    net_2 = GetNet(cn_2, wd_1)

    progress = progressbar.ProgressBar()

    for path in progress(ImagesToProcess):
        outfile = os.path.basename(path)
        outfile = os.path.join(OUT, outfile)
        #pdb.set_trace()
        PredOneImage(path, outfile, crop,  pred_f, net_1, net_2)


