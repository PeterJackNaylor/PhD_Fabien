import caffe
import numpy as np
from optparse import OptionParser
import time
import pdb
import sys


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


def Net(model_def, weights, data_batch_size, n_c=3, height=224, width=224):

    net = caffe.Net(model_def,      # defines the structure of the model
                    weights,  # contains the trained weights
                    caffe.TEST)
    net.blobs['data'].reshape(data_batch_size, n_c, height, width)
    return net

def Transformer(net):

    mu = np.array([104.00698793, 116.66876762, 122.67891434])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    return transformer


def Forward(net, img, transformer = None, layer="score"):

    if transformer is None:
        transformer = Transformer(net)

    transformed_image = transformer.preprocess('data', img)

    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    score = output[layer]

    return score

def ProcessLoss(image_blob, method="binary"):

    if method == "binary":
        
        classed = np.argmax(image_blob[0, :, :, :], axis=0)
        scores = np.unique(classed)
        def rescore(c):
            """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
            return np.where(scores == c)[0][0]
        rescore = np.vectorize(rescore)

        painted = rescore(classed)
        return painted
    
    elif method == "softmax":

        l1 = image_blob[0, 0, :, :]
        l2 = image_blob[0, 1, :, :]
        mat = np.exp(l1 - l2)
        return 1 / (1 + mat)

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






if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-p", "--path", dest="path",
                      help="Input path (raw data)")
    parser.add_option('--crop', dest="crop", default="1",
                      help="crop")
    parser.add_option("--weight", dest="weight",
                      help="Where to find the weight file")
    parser.add_option("--layer_score", dest="scorelayer",
                      help="name of score layer", default="score")
    parser.add_option("--model_def", dest="model_def",
                      help="Model definition, prototxt file")

    (options, args) = parser.parse_args()

    print " \n "
    print "Input paramters to Validation score:"
    print " \n "
    print "Input file        : | " + options.path
    print "Crop              : | " + options.crop
    print "score layer name  : | " + options.scorelayer
    print "Weight file (init): | " + options.weight
    print "Model definition  : | " + options.model_def
    start_time = time.time()


    print 'Done.'

    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
