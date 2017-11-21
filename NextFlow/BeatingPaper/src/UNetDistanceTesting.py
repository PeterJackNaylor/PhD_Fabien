import pdb
from UNetDistanceTraining import UNetDistance
from Deprocessing.Morphology import PostProcess

import math
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from UsefulFunctions.ImageTransf import ListTransform
from optparse import OptionParser
from Deprocessing.Morphology import PostProcess
from skimage.measure import label
from sklearn.metrics import f1_score
from Prediction.AJI import AJI_fast
import numpy as np



def F1compute(FP, GT):
    pr = FP.copy()
    gt = GT.copy()
    pr[pr > 0] = 1
    gt[gt > 0] = 1
    return f1_score(pr.flatten(), gt.flatten())

def AJIcompute(FP, GT):
    return AJI_fast(FP, GT)
def ACCcompute(FP, GT):
    FP[FP > 0] = 1
    GT[GT > 0] = 1
    res = np.mean((FP == GT).astype('float')) * 100
    return res


class TestModel(UNetDistance):
    def Validation(self, DG_TEST, p1, thresh):
        n_test = DG_TEST.length
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

        l = 0.
        f1 = 0
        AJI = 0
        ACC = 0
        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            #Yval = Yval / 255.
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l_tmp, pred = self.sess.run([self.loss, 
                                                self.predictions],
                                                feed_dict=feed_dict)
            pred[pred < 0] = 0
            pred = pred.astype('uint8')
            for j in range(self.BATCH_SIZE):
                FP = PostProcess(pred[j], p1, thresh)
                GT = Yval[j, :, :, 0].copy()
                GT[GT > 0] = 1
                GT = label(GT)
                f1 += F1compute(FP, GT)
                AJI += AJIcompute(FP, GT)
                ACC += ACCcompute(FP, GT)
            l += l_tmp

        l = l / n_batch
        ACC = ACC / (n_batch * self.BATCH_SIZE)
        f1 = f1 / (n_batch * self.BATCH_SIZE)
        AJI = AJI / (n_batch * self.BATCH_SIZE)
        return l, ACC, f1, AJI



if __name__== "__main__":

    parser = OptionParser()

    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")

    parser.add_option("--output", dest="output", type="string",
                      help="Where to store the output file")

    parser.add_option("--log", dest="log",
                      help="log dir")

    parser.add_option("--batch_size", dest="bs", type="int",
                      help="batch size")

    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")

    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")

    parser.add_option("--lambda", dest="p1", type="int")
    parser.add_option("--thresh", dest="thresh", type="float")

    (options, args) = parser.parse_args()


    # TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
    #                 "bladder", "colorectal", "stomach"]
    TEST_PATIENT = ["test"]
    transform_list, transform_list_test = ListTransform(n_elastic=0)

    PATH = options.path
    CROP = 1
    HEIGHT, WIDTH = 500, 500
    BATCH_SIZE = options.bs
    SAVE_DIR = options.log
    N_FEATURES = options.n_features
    MEAN_FILE = options.mean_file 


    SIZE = (HEIGHT, WIDTH)

    model = TestModel("",  LEARNING_RATE=0.001,
                           BATCH_SIZE=BATCH_SIZE,
                           IMAGE_SIZE=SIZE,
                           NUM_CHANNELS=3,
                           STEPS=100,
                           LRSTEP="10epoch",
                           N_PRINT=100,
                           LOG=SAVE_DIR,
                           SEED=42,
                           WEIGHT_DECAY=0.00005,
                           N_FEATURES=N_FEATURES,
                           N_EPOCH=10,
                           N_THREADS=1,
                           MEAN_FILE=MEAN_FILE,
                           DROPOUT=0.5)
    file_name = options.output
    f = open(file_name, 'w')
    f.write('{}, {}, {}, {}, {}, {}\n'.format("Organ", "Loss", "Acc", "F1", "AJI", "p1"))
    for organ in TEST_PATIENT:
        DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH),num=[organ],
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
        DG_TEST.SetPatient([organ])
        Loss, Acc, f1, AJI = model.Validation(DG_TEST, options.p1, options.thresh)
        f.write('{}, {}, {}, {}, {}, {}\n'.format(organ, Loss, Acc, f1, AJI, options.p1))
    f.close()
