from Nets.UNetBatchNorm_v2 import UNetBatchNorm
import math
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from UsefulFunctions.ImageTransf import ListTransform
from optparse import OptionParser


class TestModel(UNetBatchNorm):
    def Validation(self, DG_TEST):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.

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

            l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch
            return l, acc, F1, recall, precision, meanacc
        

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


    TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                    "bladder", "colorectal", "stomach"]
    transform_list, transform_list_test = ListTransform(n_elastic=0)

    PATH = options.path
    CROP = 4
    HEIGHT, WIDTH = 212, 212
    BATCH_SIZE = options.bs
    SAVE_DIR = options.log
    N_FEATURES = options.n_features
    MEAN_FILE = options.mean_file 


    SIZE = (HEIGHT, WIDTH)


    model = TestModel("",  LEARNING_RATE=0.01,
                           BATCH_SIZE=BATCH_SIZE,
                           IMAGE_SIZE=SIZE,
                           NUM_CHANNELS=3,
                           STEPS=100,
                           LRSTEP=1,
                           N_PRINT=100,
                           LOG=SAVE_DIR,
                           SEED=42,
                           WEIGHT_DECAY=0.0005,
                           N_FEATURES=N_FEATURES,
                           N_EPOCH=1,
                           N_THREADS=1,
                           MEAN_FILE=MEAN_FILE,
                           DROPOUT=0.5)
    for organ in TEST_PATIENT:

        DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH),num=[organ],
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
        DG_TEST.SetPatient([organ])

        Loss, Acc, f1, Recall, Precision, Meanacc = model.Validation(DG_TEST)
    
    file_name = options.output
    f = open(file_name, 'w')
    f.write('AJI: # {} #\n'.format(AJI))
    f.write('F1: # {} #\n'.format(f1))
    f.write('MSE: # {} #\n'.format(l)) 
    f.close()
