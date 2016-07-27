import caffe
import numpy as np
from CheckingSolvingState.OutputNet import Transformer, GetScoreVectors


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


def All(model_def, weight_file, image_array, score_layer="score"):
    net = Net(model_def, weight_file, len(image_array))
    transformer = Transformer(net)
    score = GetScoreVectors(net, image_array, transformer, score_layer)
    pred_img = np.zeros(shape=(score.shape[0], score.shape[2], score.shape[3]))
    for i in range(score.shape[0]):
        pred_img[i, :, :] = Out(score[i, :, :, :])
    return pred_img
