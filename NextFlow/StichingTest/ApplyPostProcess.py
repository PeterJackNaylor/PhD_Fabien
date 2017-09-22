from UsefulFunctions.UsefulImageConstruction import sliding_window, PredLargeImageFromNet
from optparse import OptionParser
from skimage.io import imread
import time
import caffe
import os
from WrittingTiff.createfold import GetNet
from Prediction.AJI import AJI_fast
from Deprocessing.Morphology import PostProcess
import pandas as pd

def Options():

    parser = OptionParser()

    parser.add_option('--image', dest="img", type="str",
                      help="image to analyse")
    parser.add_option('--gt', dest="gt", type="str",
                      help="ground truth")
    parser.add_option('--clearborder', dest="clearborder", type="str",
                      help="method to use RemoveBorder etc")
    parser.add_option('--method', dest="method", type="str",
                      help="How to aggregate the probability results (max, median, avg)")
    parser.add_option('--stepsize', dest="stepsize", type="int",
                      help="stepsize for analysing the whole image")
    parser.add_option('--lambda', dest="lambda_", type="int",
                      help="lambda value for the post processing")
    parser.add_option('--output', dest="output", type="str",
                      help="how to name the csv")
    parser.add_option('--wd', dest="wd", type="str",
                      help="work directory for the net")
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    options = Options()

    caffe.set_mode_cpu()
    cn_1 = "FCN_0.01_0.99_0.005"
    wd_1 = options.wd #"/share/data40T_v2/Peter/pretrained_models"
    net_1 = GetNet(cn_1, wd_1)

    image = imread(options.image)
    GT = imread(options.gt)
    start = time.time()
    prob, bin_map, thresh = PredLargeImageFromNet(net_1, image, options.stepsize, 
                          (224, 224), removeFromBorder=1, 
                          method=options.method, param=options.lambda_, 
                          ClearBorder=options.clearborder)

    bin_image = PostProcess(prob, options.lambda_, thresh=thresh)
    end = time.time() - start

    count = 0
    for obj in sliding_window(image, options.stepsize, (224,224)):
        count += 1

    time_to_convert = end / count
    score = AJI_fast(GT, bin_image)
    dic = {'method':[options.method,],
           'clearborder':[options.clearborder,],
           'stepsize':[options.stepsize,],
           'AJI':[score,],
           'lambda':[options.lambda_,],
           'time':[time_to_convert,],
           'n_img':[count,]}
    df = pd.DataFrame(dic)
    df.to_csv(options.output, index=False)
