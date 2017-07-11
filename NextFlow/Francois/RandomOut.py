from optparse import OptionParser
import caffe
from Deprocessing.Morphology import PostProcess
from UsefulFunctions.UsefulImageConstruction import Contours
from WrittingTiff.FrancoisRadvani import pred_f, crop
from Deprocessing.Transfer import ChangeEnv
import os
from WrittingTiff.createfold import GetNet
from scipy.misc import imread, imsave
import numpy as np
from skimage.morphology import remove_small_objects
from UsefulFunctions.RandomUtils import CheckOrCreate
import pdb
import time

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--file_name", dest="file_name",
                      help="Input file (raw data)")

    parser.add_option("--output", dest="output",
                      help="Where to store the outputs")
    parser.add_option('--env', dest="env", 
                      help="Where the annotated images are")
    parser.add_option('--wd', dest="wd", 
                      help="working directory")

    (options, args) = parser.parse_args()


    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.005"
    wd_1 = options.wd #"/share/data40T_v2/Peter/pretrained_models"
    cn_2 = "DeconvNet_0.01_0.99_0.0005"
    wd_2 = wd_1
    net_1 = GetNet(cn_1, wd_1)
    net_2 = GetNet(cn_2, wd_1)


    step_size = 100 
    param_ws = 10
    ClearSmallObjects = False

    base = os.path.basename(options.file_name).replace(".png", "")
    CheckOrCreate(base)
    base = base + "/" + base

    base_img = os.path.join(options.output, base + "_RAW.png")
    base_seg = os.path.join(options.output, base + "_seg.png")
    base_prob = os.path.join(options.output, base + "_prob.png")
    base_bin = os.path.join(options.output, base + "_bin.png")

    image = imread(options.file_name)[:,:,0:3]
    image = crop(image)

    for method in ['avg']: #["avg", "max"]:
        base_img = os.path.join(options.output, base + "_" + method + "_RAW.png")
        base_seg = os.path.join(options.output, base + "_" + method + "_seg.png")
        base_prob = os.path.join(options.output, base + "_" + method + "_prob.png")
        base_bin = os.path.join(options.output, base + "_" + method + "_bin.png")
        diff_time = time.time()
        prob, thresh = pred_f(image, net_1, net_2, stepSize=step_size, windowSize=(224, 224), param=7, border = 1,
                      borderImage = "Classic", method = method)
        diff_time = diff_time - time.time()
        bin_image = PostProcess(prob, param_ws, thresh=thresh)
        if ClearSmallObjects:
            bin_image = remove_small_objects(bin_image, ClearSmallObjects)

        ContourSegmentation = Contours(bin_image)
        x_, y_ = np.where(ContourSegmentation > 0)
        seg_image = image.copy()
        seg_image[x_,y_,:] = np.array([0,0,0])

        imsave(base_img, image)
        imsave(base_bin, bin_image)
        imsave(base_seg, seg_image)
        imsave(base_prob, prob)

    for borderImage in []:# ["Classic", "RemoveBorderObjects", "RemoveBorderWithDWS", "Reconstruction"]:
        method = borderImage
        base_img = os.path.join(options.output, base + "_" + method + "_RAW.png")
        base_seg = os.path.join(options.output, base + "_" + method + "_seg.png")
        base_prob = os.path.join(options.output, base + "_" + method + "_prob.png")
        base_bin = os.path.join(options.output, base + "_" + method + "_bin.png")
        prob, thresh = pred_f(image, net_1, net_2, stepSize=step_size, windowSize=(224, 224), param=7, border = 1,
                      borderImage = borderImage, method = "max")
        bin_image = PostProcess(prob, param_ws, thresh=thresh)
        if ClearSmallObjects:
            bin_image = remove_small_objects(bin_image, ClearSmallObjects)

        ContourSegmentation = Contours(bin_image)
        x_, y_ = np.where(ContourSegmentation > 0)
        seg_image = image.copy()
        seg_image[x_,y_,:] = np.array([0,0,0])

        imsave(base_img, image)
        imsave(base_bin, bin_image)
        imsave(base_seg, seg_image)
        imsave(base_prob, prob)

    print "Average time per: \n "
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600)/ 60, diff_time % 60)