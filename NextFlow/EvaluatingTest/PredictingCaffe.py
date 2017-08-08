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
from sklearn.metrics import confusion_matrix
from os.path import join, basename


if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--image", dest="image",
                      help="Input file (raw data)")
    parser.add_option("--anno", dest="anno",
                      help="corresponding annotation file")
    parser.add_option("--output", dest="output",
                      help="Where to store the outputs")
    parser.add_option('--env', dest="env", 
                      help="Where the annotated images are")
    parser.add_option('--wd', dest="wd", 
                      help="working directory")
    parser.add_option('--param', dest="param", type='int',  
                      help="parameter for the post-process")
    parser.add_option('--stepSize', dest="stepSize", type='int', 
                      help="stepSize for the slidding window")
    parser.add_option('--net_1', dest="net_1", 
                      help="net_1")
    parser.add_option('--net_2', dest="net_2", 
                      help="net_2")
    (options, args) = parser.parse_args()


    caffe.set_mode_cpu()

    wd = options.wd 

    net_1 = GetNet(options.net_1, wd)
    
    net_2 = GetNet(options.net_2, wd) if options.net_2 is not None else None


    ClearSmallObjects = 50

    output = options.output
    CheckOrCreate(output)
    image = imread(options.image)
    anno = imread(options.anno)
    dic = pred_f(image, net_1, net_2, stepSize=options.stepSize, windowSize=(224, 224), param=options.param, border = 1,
                  borderImage = "Reconstruction", method = "max", return_all = True)
    
    for model in dic.keys():
        prob, thresh = dic[model]
        if model == "model1":
            model = options.net_1
        elif model == "model2":
            model = options.net_2

        output_mod = join(output, model + "_model")
        CheckOrCreate(output_mod)
        base_img = join(output_mod, "Input.png")
        base_label = join(output_mod, "Label.png")
        base_seg = join(output_mod, "Segmented.png")
        base_prob = join(output_mod, "Prob.png")
        base_bin = join(output_mod, "Bin.png")


        bin_image = PostProcess(prob, options.param, thresh=thresh)
        if ClearSmallObjects:
            bin_image = remove_small_objects(bin_image, ClearSmallObjects)

        ContourSegmentation = Contours(bin_image)
        x_, y_ = np.where(ContourSegmentation > 0)
        seg_image = image.copy()
        seg_image[x_,y_,:] = np.array([0,0,0])

        bin_image[bin_image > 0] = 255	
        imsave(base_label, anno)
        imsave(base_img, image)
        imsave(base_bin, bin_image)
        imsave(base_seg, seg_image)
        imsave(base_prob, prob)

        cm = confusion_matrix(anno.flatten(), bin_image.flatten(), labels=[0, 255]).astype(np.float)
        TP = cm[1, 1]
        TN = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        precision = float(TP) / (TP + FP) 
        recall = float(TP) / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        acc = float(TP + TN) / (TP + TN + FP + FN)



        file_name = join(output, model + "_model", "Characteristics.txt")
        f = open(file_name, 'w')
        f.write('TP: # {} #\n'.format(TP))
        f.write('TN: # {} #\n'.format(TN))
        f.write('FN: # {} #\n'.format(FN))
        f.write('FP: # {} #\n'.format(FP))
        f.write('Precision: # {} #\n'.format(precision))
        f.write('Recall: # {} #\n'.format(recall))
        f.write('F1: # {} #\n'.format(F1))
        f.write('ACC: # {} #\n'.format(acc)) 
        f.close()

