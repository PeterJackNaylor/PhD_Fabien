from optparse import OptionParser
from Prediction.AJI import AJI_fast
from skimage.measure import label
from Deprocessing.Morphology import PostProcess
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import jaccard_similarity_score, f1_score
from sklearn.metrics import recall_score, precision_score
from UsefulFunctions.RandomUtils import add_contours, color_bin
from os.path import join
from skimage.io import imsave

def GetOptions():

    parser = OptionParser()
    parser.add_option("--tf_record", dest="TFRecord", type="string", default="",
                      help="Where to find the TFrecord file")
    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")
    parser.add_option("--size_train", dest="size_train", type="int",
                      help="size of the input image to the network")
    parser.add_option("--log", dest="log",
                      help="log dir")
    parser.add_option("--learning_rate", dest="lr", type="float", default=0.01,
                      help="learning_rate")
    parser.add_option("--batch_size", dest="bs", type="int", default=1,
                      help="batch size")
    parser.add_option("--epoch", dest="epoch", type="int", default=1,
                      help="number of epochs")
    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")
    parser.add_option("--weight_decay", dest="weight_decay", type="float", default=0.00005,
                      help="weight decay value")
    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5, help="dropout value to apply to the FC layers.")
    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")
    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")
    parser.add_option('--crop', dest="crop", type=int, default=4,
                      help="crop size depending on validation/test/train phase.")
    parser.add_option('--split', dest="split", type="str",
                      help="validation/test/train phase.")
    parser.add_option('--p1', dest="p1", type="int",
                      help="1st input for post processing.")
    parser.add_option('--p2', dest="p2", type="float",
                      help="2nd input for post processing.")

    parser.add_option('--iters', dest="iters", type="int")
    parser.add_option('--seed', dest="seed", type="int")
    parser.add_option('--size_test', dest="size_test", type="int")
    parser.add_option('--restore', dest="restore", type="str")
    parser.add_option('--save_path', dest="save_path", type="str", default=".")
    parser.add_option('--type', dest="type", type ="str",
                       help="Type for the datagen")  
    parser.add_option('--UNet', dest='UNet', action='store_true')
    parser.add_option('--no-UNet', dest='UNet', action='store_false')
    parser.add_option('--output', dest="output", type="str")

    (options, args) = parser.parse_args()

    return options

def ComputeMetrics(prob, batch_labels, p1, p2, rgb=None, save_path=None, ind=0):
    GT = label(batch_labels.copy())
    PRED = PostProcess(prob, p1, p2)
    lbl = GT.copy()
    pred = PRED.copy()
    aji = AJI_fast(lbl, pred)
    lbl[lbl > 0] = 1
    pred[pred > 0] = 1 
    l, p = lbl.flatten(), pred.flatten()
    acc = accuracy_score(l, p)
    roc = roc_auc_score(l, p)
    jac = jaccard_similarity_score(l, p)
    f1 = f1_score(l, p)
    recall = recall_score(l, p)
    precision = precision_score(l, p)
    if rgb is not None:
        xval_n = join(save_path, "xval_{}.png").format(ind)
        yval_n = join(save_path, "yval_{}.png").format(ind)
        prob_n = join(save_path, "prob_{}.png").format(ind)
        pred_n = join(save_path, "pred_{}.png").format(ind)
        c_gt_n = join(save_path, "C_gt_{}.png").format(ind)
        c_pr_n = join(save_path, "C_pr_{}.png").format(ind)
        ## CHECK PLOT FOR PROB AS IT MIGHT BE ILL ADAPTED

        imsave(xval_n, rgb)
        imsave(yval_n, color_bin(GT))
        imsave(prob_n, prob)
        imsave(pred_n, color_bin(PRED))
        imsave(c_gt_n, add_contours(rgb, GT))
        imsave(c_pr_n, add_contours(rgb, PRED))

    return acc, roc, jac, recall, precision, f1, aji
