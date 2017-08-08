from optparse import OptionParser
import tensorflow as tf 
from scipy.misc import imread, imsave
from os.path import join, basename
import numpy as np
from UsefulFunctions.UsefulImageConstructionTF import PredLargeImageFromNet, Contours
from UNetBatchNorm_v2 import UNetBatchNorm
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix
from Deprocessing.Morphology import PostProcess
from UsefulFunctions.RandomUtils import CheckOrCreate
from glob import glob


CUDA_NODE = 0
HEIGHT = 212 
WIDTH = 212
CROP = 4
PATH = '/Users/naylorpeter/Documents/Histopathologie/NeerajKumar/ForDatagen'
BATCH_SIZE = 1
S = True

def Options():

    parser = OptionParser()

    parser.add_option('-i', dest="i", type="str",
                      help="input image")
    parser.add_option('-a', dest="a", type="str",
                      help="annotation image")
    parser.add_option('-f', dest="folder", type="str",
                      help="folder to find the latest meta data")
    parser.add_option('--mean_file', dest="mean_file", type="string",
                      help="mean_file folder")
    parser.add_option('--param', dest="param", type='int', default=10,
                      help="Value for the post-processing")  
    (options, args) = parser.parse_args()
    return options


def parse_meta_file(string):
    return int(string.split('ckpt-')[1].split('.'))

def find_latest(folder):
    all_meta = glob(join(folder, "*.meta"))
    atm = parse_meta_file(all_meta[0])
    res = all_meta[0]
    for met in all_meta:
        tmp = parse_meta_file(met)
        if tmp > atm:
            res = met
            atm = tmp
    return met



META = "/Users/naylorpeter/Desktop/Experiences/UNet/32_0.00005_0.0001/model.ckpt-2500.meta"



if __name__ == '__main__':
    options = Options()

    img = imread(options.i)
    anno = imread(options.a)

    x, y, c = (212, 212, 3)

    stepSize = x
    windowSize = (x + 184, y + 184)
    META = find_latest(options.f)
    n_features = int(basename(options.f).split('_')[0])

    model = UNetBatchNorm("f",
                          BATCH_SIZE=1, 
                          IMAGE_SIZE = (x, y),
                          NUM_CHANNELS=c, 
                          NUM_LABELS=2,
                          N_FEATURES=n_features)

    prob_map, bin_map, threshold = PredLargeImageFromNet(model, META, img, stepSize, windowSize, removeFromBorder=0, 
                                                         method="max", param=0, ClearBorder="Classic",
                                                         threshold = 0.5, UNet=True, MEAN_FILE=options.mean_file) 
    prob_map = 1 - prob_map
    bin_map[ bin_map > 0 ] = 1
    bin_map[ bin_map == 0 ] = 255
    bin_map[ bin_map == 1 ] = 0
    cm = confusion_matrix(anno.flatten(), bin_map.flatten(), labels=[0, 255]).astype(np.float)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    precision = float(TP) / (TP + FP) 
    recall = float(TP) / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    acc = float(TP + TN) / (TP + TN + FP + FN)

    PP = PostProcess(prob_map, param=options.param)


    CellCont = Contours(PP)
    contour_rgb = img[92:-92, 92:-92].copy()
    contour_rgb[CellCont > 0] = np.array([0, 0, 0])

    output = basename(options.i)
    CheckOrCreate(output)

    imsave(join(output, "Input.png"), img)
    imsave(join(output, "Segmented.png"), contour_rgb)
    imsave(join(output, "Prob.png"),img_as_ubyte(prob_map))
    imsave(join(output, "Bin.png"), bin_map)

    file_name = join(output, "Characteristics.txt")
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