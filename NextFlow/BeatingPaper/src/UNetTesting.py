from Nets.UNetBatchNorm_v2 import UNetBatchNorm
import pdb
import math
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from UsefulFunctions.ImageTransf import ListTransform
from optparse import OptionParser
from UNetTraining import unet_diff
import tensorflow as tf
import numpy as np
from skimage.measure import label

from Prediction.AJI import AJI_fast
	
from Deprocessing.Morphology import PostProcess


class TestModel(unet_diff):
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=2,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        STEPS=2000,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True,
        WEIGHT_DECAY=0.00005,
        LOSS_FUNC=tf.nn.l2_loss,
        N_FEATURES=16,
        N_EPOCH=1,
        N_THREADS=1,
        MEAN_FILE=None,
        DROPOUT=0.5):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.N_FEATURES = N_FEATURES
#        self.NUM_TEST = NUM_TEST
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED
        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, 3])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False

        self.sess = tf.InteractiveSession()

        self.sess.as_default()
        
        self.var_to_reg = []
        self.var_to_sum = []
        self.TF_RECORDS = TF_RECORDS
        self.init_queue(TF_RECORDS)

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.DEBUG = DEBUG
        self.loss_func = LOSS_FUNC
        self.weight_decay = WEIGHT_DECAY
        self.Saver()

    def Validation(self, DG_TEST, p1, p2):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc, AJI = 0., 0., 0., 0., 0., 0., 0.

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                # Yval[Yval > 0] = 1.
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, prob     = self.sess.run([self.loss, 
                                                                                            self.accuracy, self.F1,
                                                                                            self.recall, self.precision,
                                                                                            self.MeanAcc, self.train_prediction],
                                                                                            feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                F1 += F1_tmp
                recall += recall_tmp
                precision += precision_tmp
                meanacc += meanacc_tmp
                for j in range(self.BATCH_SIZE):
                    FP = PostProcess(prob[j,:,:,0], p1, p2)
                    GT = Yval[j, :, :, 0]
                    GT[GT > 0] = 1
                    GT = label(GT)
                    AJI += AJI_fast(FP, GT)
                AJI = AJI / self.BATCH_SIZE
            l, acc, F1, recall, precision, meanacc, AJI = np.array([l, acc, F1, recall, precision, meanacc, AJI]) / n_batch
            return l, acc, F1, recall, precision, meanacc, AJI


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
    f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format("Organ", "Loss", "Acc", "F1", "Recall", "Precision", "Meanacc", "AJI"))
    for organ in TEST_PATIENT:
        DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH),num=[organ],
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
        DG_TEST.SetPatient([organ])
        Loss, Acc, f1, Recall, Precision, Meanacc, AJI = model.Validation(DG_TEST, options.p1, options.thresh)
        f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(organ, Loss, Acc, f1, Recall, Precision, Meanacc, AJI, options.p1))
    f.close()
