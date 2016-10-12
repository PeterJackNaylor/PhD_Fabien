import caffe
import numpy as np
from CheckingSolvingState.OutputNet import Transformer, GetScoreVectors
import sys

from skimage.morphology import watershed
from skimage import measure
from scipy import ndimage as ndi


def Net(model_def, weights, data_batch_size, height=512, width=512):
    net = caffe.Net(model_def,      # defines the structure of the model
                    weights,  # contains the trained weights
                    caffe.TEST)
    net.blobs['data'].reshape(data_batch_size, 3, height, width)
    return net


def Out(loss_image):
    classed = np.argmax(loss_image, axis=0)

    names = dict()
    all_labels = ["0: Background"] + ["1: Cell"]
    scores = np.unique(classed)
    labels = [all_labels[s] for s in scores]
    num_scores = len(scores)

    def rescore(c):
        """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
        return np.where(scores == c)[0][0]
    rescore = np.vectorize(rescore)

    painted = rescore(classed)
    return painted


def All(model_def, weight_file, image_array, score_layer="score", height=512, width=512):
    net = Net(model_def, weight_file, len(
        image_array), height=height, width=width)
    transformer = Transformer(net)
    score = GetScoreVectors(net, image_array, transformer, score_layer)
    pred_img = np.zeros(shape=(score.shape[0], score.shape[2], score.shape[3]))
    for i in range(score.shape[0]):
        pred_img[i, :, :] = Out(score[i, :, :, :])
    return pred_img


def Preprocessing(image_RGB):
    in_ = np.array(image_RGB, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array([104.00698793, 116.66876762, 122.67891434])
    in_ = in_.transpose((2, 0, 1))
    return in_


def Deprocessing4Visualisation(image_blob):
    # pdb.set_trace()
    new = image_blob[0, :, :, :].transpose(1, 2, 0).copy()
    new = new[:, :, ::-1]
    new += np.array([104.00698793, 116.66876762, 122.67891434])
    new[new > 255] = 255
    new[new < 0] = 0
    new = new.astype(np.uint8)

    return new


def DeprocessingLabel(image_blob):
    return image_blob[0, 0, :, :]


def OutputNet(image_blob, method="binary"):

    l1 = image_blob[0, 0, :, :]
    l2 = image_blob[0, 1, :, :]

    if method == "binary":
        classed = np.argmax(image_blob[0, :, :, :], axis=0)
        names = dict()
        all_labels = ["0: Background"] + ["1: Cell"]
        scores = np.unique(classed)
        labels = [all_labels[s] for s in scores]
        num_scores = len(scores)

        def rescore(c):
            """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
            return np.where(scores == c)[0][0]
        rescore = np.vectorize(rescore)

        painted = rescore(classed)
        return painted
    elif method == "softmax":
        mat = np.exp(l1 - l2)
        return 1 / (1 + mat)
#        expl1 = np.exp(-l1)
#        expl2 = np.exp(-l2)
#        mat = expl1 / (expl1 + expl2)
        return mat
    elif method == "log":
        min_to_add = min(np.min(l1), np.min(l2))
        test1 = (l1 + min_to_add + 1).astype(np.dtype('float64'))
        test2 = (l2 + min_to_add + 1).astype(np.dtype('float64'))
        mat = test1 / (test1 + test2)
        return mat
    elif method == "normalize":
        maxint = sys.maxint - 100
        res = np.zeros(l1.shape[0] * l2.shape[1] * 2)
        res[0:(l1.shape[0] * l2.shape[1])] = l1.flatten()
        res[(l1.shape[0] * l2.shape[1])            :(l1.shape[0] * l2.shape[1] * 2)] = l2.flatten()
        mu = np.mean(res)
        sig = np.std(res)

        mat = np.exp((l2 - mu) / sig - (l1 - mu) / sig)
        mat[mat > maxint] = maxint
        return 1 / (1 + mat)


def PrepareProb(img, convertuint8=True, inverse=True):
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img


def HreconstructionErosion(prob_img, h):

    def making_top_mask(x, lamb=h):
        if 255 >= x + lamb:
            return x + lamb
        else:
            return 255
    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(seed, mask, method='erosion')
    return recons


def find_maxima(img, convertuint8=False, inverse=False):
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    return recons - img


def ProbDeprocessing(prob_image, bin_image, param, method="ws_recons"):
    if methd == "ws_recons":
        lamb = param

        Probs_inv = PrepareProb(prob_image)
        Hrecons = HreconstructionErosion(Probs_inv, lamb)
        markers_Probs_inv = find_maxima(Hrecons)
    # you have to give different seed values for each connected componant
        markers_Probs_inv = measure.label(markers_Probs_inv)
        ws_labels = watershed(Hrecons, markers_Probs_inv, mask=bin_image)
        new_ws_labels = ws_labels.copy()
        for val_m in range(1, np.max(ws_labels) + 1):
            temp = ws_labels.copy()
            temp[temp != val_m] = 0
            temp[temp == val_m] = 1
            Contours = dilation(temp, square(2)) - erosion(temp, square(2))
            new_ws_labels[Contours == 1] = 0

        new_ws_labels[new_ws_labels > 0] = 1
        return new_ws_labels
