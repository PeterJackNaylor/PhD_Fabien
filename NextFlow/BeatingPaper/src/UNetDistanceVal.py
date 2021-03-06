import pdb
from UNetDistanceTraining import UNetDistance
from Deprocessing.Morphology import PostProcess
import pandas as pd
import math
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from UsefulFunctions.ImageTransf import ListTransform
from optparse import OptionParser
from Deprocessing.Morphology import PostProcess
from skimage.measure import label
from sklearn.metrics import f1_score
from Prediction.AJI import AJI_fast
import numpy as np
from os.path import join
import glob
from UsefulFunctions.RandomUtils import CheckOrCreate, color_bin, add_contours
from skimage.io import imsave

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
    res = np.mean((FP == GT).astype('float'))
    return res


class TestModel(UNetDistance):
    def Validation(self, DG_TEST, p1, thresh, save_folder):
        n_test = DG_TEST.length
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

        l = []
        f1 = []
        AJI = []
        ACC = []
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
                xval_name = join(save_folder, "X_val_{}_{}.png".format(i, j))
                yval_name = join(save_folder, "Y_val_{}_{}.png".format(i, j))
                pred_name = join(save_folder, "pred_{}_{}.png".format(i, j))
                pred_bin_name = join(save_folder, "predbin_{}_{}.png".format(i, j))
                cont_true_name = join(save_folder, "contpred_{}_{}.png".format(i, j))
                cont_pred_name = join(save_folder, "conttrue_{}_{}.png".format(i, j))
                imsave(pred_name, pred[j])
                imsave(xval_name, (Xval[j,92:-92,92:-92] + np.load('mean_file.npy')).astype('uint8'))
                FP = PostProcess(pred[j], p1, thresh)
                imsave(pred_bin_name, color_bin(FP))
                GT = Yval[j, :, :, 0].copy()
                GT[GT > 0] = 1
                GT = label(GT)
                f1.append(F1compute(FP, GT))
                AJI.append(AJIcompute(FP, GT))
                ACC.append(ACCcompute(FP.copy(), GT.copy()))
                imsave(yval_name, color_bin(GT))
                rgb = (Xval[j,92:-92,92:-92] + np.load('mean_file.npy')).astype('uint8')
                imsave(cont_true_name, add_contours(rgb, GT))
                imsave(cont_pred_name, add_contours(rgb, FP))

            l.append(l_tmp)

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
    parser.add_option("--test_res", dest="test_res", type="str")
    parser.add_option("--save_sample", dest="save_sample", type="str")
    (options, args) = parser.parse_args()


    

    TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                    "bladder", "colorectal", "stomach"]
    transform_list, transform_list_test = ListTransform(n_elastic=0)

    PATH = options.path
    CROP = 1
    HEIGHT, WIDTH = 996, 996
    BATCH_SIZE = options.bs
    SAVE_DIR = glob.glob(join(options.log, "*_0.*"))[0]
    N_FEATURES = options.n_features
    MEAN_FILE = options.mean_file 

    #### Figuring out p1:
    if options.p1 is None:    
        table = pd.read_csv(options.test_res)
        options.p1 = table.ix[table[' F1'].argmax(), " p1"]
    ####

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
    f.write('{},{},{},{},{},{},{}\n'.format("ORGAN", "NUMBER", "Loss", "ACC", "F1", "AJI", "LAMBDA"))
    
    CheckOrCreate(options.save_sample)
    count = 0
    for organ in TEST_PATIENT:
        DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH),num=[organ],
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
        DG_TEST.SetPatient([organ])
        save_organ = join(options.save_sample, organ)
        CheckOrCreate(save_organ)
        Loss, Acc, f1, AJI = model.Validation(DG_TEST, options.p1, options.thresh, save_organ)
        for i in range(len(Loss)):
            f.write('{},{},{},{},{},{},{},{}\n'.format(count, organ, i, Loss[i], Acc[i], f1[i], AJI[i], options.p1))
            count += 1
    f.close()
