from optparse import OptionParser
import tensorflow as tf 
from scipy.misc import imread, imsave
from os.path import join, basename
import numpy as np
from UsefulFunctions.UsefulImageConstructionTF import PredLargeImageFromNet, Contours
from NewStuff.UNetBatchNorm_v2 import UNetBatchNorm
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix
from Deprocessing.Morphology import PostProcess
from UsefulFunctions.RandomUtils import CheckOrCreate
from glob import glob
import pdb
from Data.DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform

class bettermodel(UNet):
    def Validation(self, DG_TEST):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc, AJI = 0., 0., 0., 0., 0., 0., 0.

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, s = self.sess.run([self.loss, 
                                                                                            self.accuracy, self.F1,
                                                                                            self.recall, self.precision,
                                                                                            self.MeanAcc,
                                                                                            self.merged_summary], feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                F1 += F1_tmp
                recall += recall_tmp
                precision += precision_tmp
                meanacc += meanacc_tmp
                AJI += 0.
            l, acc, F1, recall, precision, meanacc, AJI = np.array([l, acc, F1, recall, precision, meanacc, AJI]) / n_batch
            return l, acc, F1, recall, precision, meanacc, AJI
def Options():

    parser = OptionParser()

    parser.add_option('-path', dest="path", type="str",
                      help="path to folder image")
    parser.add_option('-output', dest="output", type="str",
                      help="output folder")
    parser.add_option('-f', dest="folder", type="str",
                      help="folder to find the latest meta data")
    parser.add_option('--mean_file', dest="mean_file", type="string",
                      help="mean_file folder")
    (options, args) = parser.parse_args()
    return options


def parse_meta_file(string):
    return int(string.split('ckpt-')[1].split('.')[0])

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




if __name__ == '__main__':
    options = Options()

    ### PUT DATAGEN
    MEAN_FILE = options.mean_file 
    transform_list, transform_list_test = ListTransform()
    output = options.output

    test_patient = ["141549", "162438"]
    x, y, c = (212, 212, 3)

    DG_TEST  = DataGenRandomT(options.path, split="test", crop = 4, size=(x, y), 
                       transforms=transform_list_test, seed_=42, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)


    stepSize = x
    windowSize = (x + 184, y + 184)
    META = find_latest(options.folder)

    n_features = int(basename(options.folder).split('_')[0])

    model = bettermodel("f",
                          BATCH_SIZE=1, 
                          IMAGE_SIZE = (x, y),
                          NUM_CHANNELS=c, 
                          NUM_LABELS=2,
                          N_FEATURES=n_features)

    l, acc, F1, recall, precision, meanacc, AJI = model.Validation(DG_TEST)

    file_name = join(output, "Characteristics.txt")
    f = open(file_name, 'w')
    f.write('AJI: # {} #\n'.format(AJI))
    f.write('Mean acc: # {} #\n'.format(meanacc))
    f.write('Precision: # {} #\n'.format(precision))
    f.write('Recall: # {} #\n'.format(recall))
    f.write('F1: # {} #\n'.format(F1))
    f.write('ACC: # {} #\n'.format(acc)) 
    f.close()
