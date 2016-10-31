import cPickle as pkl
from UsefulFunctions.usefulPloting import Contours
import caffe
import matplotlib.pylab as plt
from sys import maxint
import FIMM_histo.deconvolution as deconv
import pdb
from DataGen import DataGen


class DataLayerPeter(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for FCN semantic segmentation.

        example

        params = dict(dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val", classifier_name="FCN32")
        """

        # config
        params = eval(self.param_str)
        self.split = params['split']
        self.classifier_name = params['classifier_name']
        self.mean = np.array(params['mean'])

        self.normalize = params.get('normalize', True)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size', 1)

        self.datagen = pkl.load(open(params['datagen'], 'rb'))
        self.datagen.ReLoad(self.split)
        # pdb.set_trace()
        if not hasattr(self.datagen, "Weight"):
            self.datagen.Weight = False
        if not self.datagen.Weight:
            n_tops = 2
            n_tops_str = "two"
        else:
            n_tops = 3
            n_tops_str = "three"
        # two tops: data and label
        if len(top) != n_tops:
            raise Exception(
                "Need to define {} tops: data and label.".format(n_tops_str))
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.key = self.datagen.RandomKey(False)

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.key = self.datagen.RandomKey(True)

    def reshape(self, bottom, top):
        # load image + label image pair
        IsTheirWeights = self.datagen.Weight

        if self.batch_size == 1:
            if not IsTheirWeights:
                self.data, self.label = self.loadImageAndGT(self.key)
            else:
                self.data, self.label, self.weight = self.loadWithWeight(
                    self.key)
                top[2].reshape(self.batch_size, *self.weight.shape)

            top[0].reshape(self.batch_size, *self.data.shape)
            top[1].reshape(self.batch_size, *self.label.shape)

        else:
            if not IsTheirWeights:
                data, label = self.loadImageAndGT(self.key)
            else:
                data, label, weight = self.loadWithWeight(self.key)

            x, y, z = data.shape
            x_l, y_l, z_l = label.shape

            self.data = np.zeros(shape=(self.batch_size, x, y, z))
            self.label = np.zeros(shape=(self.batch_size, x_l, y_l, z_l))
            if IsTheirWeights:
                self.weight = np.zeros(
                    shape=(self.batch_size, x_l, y_l, z_l), dtype=np.float32)
                self.weight[0] = weight
            self.data[0], self.label[0] = data, label

            for i in range(1, self.batch_size):
                self.Nextkey()
                if not IsTheirWeights:
                    self.data[i], self.label[i] = self.loadImageAndGT(self.key)
                else:
                    self.data[i], self.label[i], self.weight[
                        i] = self.loadWithWeight(self.key)
                # reshape tops to fit (leading 1 is for batch dimension)
            top[0].reshape(self.batch_size, *data.shape)
            top[1].reshape(self.batch_size, *label.shape)

            if IsTheirWeights:
                top[2].reshape(self.batch_size, *weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        self.label[self.label > 0] = 1
        top[1].data[...] = self.label
        if self.datagen.Weight:
            weight = self.weight
            if 0 in weight:
                weight[weight > 0] = 1
            top[2].data[...] = weight
        # pick next input
        if len(np.unique(top[1].data[...])) != 2:
            print np.unique(top[1].data[...])
            if 1 not in np.unique(top[1].data[...]):
                print np.unique(top[1].data[...])
        # pdb.set_trace()
	from scipy.misc import imsave
	
	from ShortPrediction import Deprocessing4Visualisation, DeprocessingLabel
	imsave('/data/users/pnaylor/temp/'+str(self.key)+"rgb.png",Deprocessing4Visualisation(top[0].data[...]))
	imsave('/data/users/pnaylor/temp/'+str(self.key)+"bin.png",DeprocessingLabel(top[1].data[...]))
	try:
	    imsave('/data/users/pnaylor/temp/'+str(self.key)+"wgt.png",DeprocessingLabel(top[2].data[...]))
	except:
	    pass
        self.Nextkey()
    def backward(self, top, propagate_down, bottom):
        pass

    def Nextkey(self):
        if self.random:
            self.key = self.datagen.NextKeyRandList(self.key)
        else:
	    if self.split != "test":
                print 'this is not random!'
            self.key = self.datagen.NextKey(self.key)

    def PrepareImg(self, img):
        in_ = np.array(img, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        if len(in_.shape) == 4:
            in_ = in_[:, :, :, 0]
        return in_

    def Prepare2DImage(self, img):
        if len(img) == 3:
            img = img.transpose((2, 0, 1))
        else:
            img = img[np.newaxis, ...]
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        return img

    def loadImageAndGT(self, key):
        im, label = self.datagen[key]
        in_ = self.PrepareImg(im)
        label = self.Prepare2DImage(label)
        if self.normalize:
            label[label > 0] = 1
        return in_, label

    def loadWithWeight(self, key):
        im, label, weight = self.datagen[key]
        in_ = self.PrepareImg(im)
        label = self.Prepare2DImage(label)
        weight = self.Prepare2DImage(weight)
        if self.normalize:
            label[label > 0] = 1
        # pdb.set_trace()
        return in_, label, weight


import glob
import numpy as np
import random
from sklearn.cross_validation import KFold
from scipy import misc
import nibabel as ni
import pdb
from itertools import chain
import sys


def duplicate_channel(blob, n_c):
    return np.tile(blob, (n_c, 1, 1, 1)).transpose((1, 0, 2, 3))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    n_c = x.shape[1]
    out = e_x / duplicate_channel(e_x.sum(axis=1), n_c)
    return out


def Duplicate(label_blob, inverse=False):
    try:
        batch, sizex, sizey = label_blob.shape
    except:
        pdb.set_trace()
    new_blob = np.zeros(shape=(batch, 2, sizex, sizey))
    for i in range(batch):
        new_blob[i, 0] = label_blob[i, 0]
        if inverse:
            new_blob[i, 1] = 1 - label_blob[i, 1]
        else:
            new_blob[i, 1] = label_blob[i, 1]
    return new_blob


def log(loss_blob):
    epsilon = 0.00001
    if 0 in loss_blob:
        loss_blob[loss_blob == 0] = epsilon
    return -np.log(loss_blob)


class WeigthedLossLayer(caffe.Layer):
    """

    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception(
                "Need three inputs to weighted softmax. 3: weight image")

    def reshape(self, bottom, top):
       # check input dimensions match
        if bottom[0].count != 2 * bottom[1].count:
            raise Exception("Inputs 0 and 1 must have the same dimension.")
        if bottom[1].count != bottom[2].count:
            raise Exception("Inputs 1 and 2 must have the same dimension.")
        if bottom[0].count != 2 * bottom[2].count:
            raise Exception("Inputs 0 and 2 must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        if np.isnan(bottom[0].data[...][0, 0, 0, 0]):
            def replace_nan(a, b):
                if np.isnan(a):
                    return b
                else:
                    return a
            v_replace_nan = np.vectorize(replace_nan)
            bottom[0].data[...] = v_replace_nan(bottom[0].data[...], 0)
        # pdb.set_trace()
        blob_score = bottom[0].data[...]
        label_batch = bottom[1].data[...].sum(axis=1)
        if len(label_batch.shape) > 4:
            label_batch = label_batch.sum(axis=4)
        weight_batch = bottom[2].data[...].sum(axis=1)
        if len(weight_batch.shape) > 3:
            weight_batch = weight_batch.sum(axis=3)
        inv_label_batch = (1 - label_batch)
        loss_batch = softmax(blob_score)
        self.diff[:, 0, :, :] = (
            loss_batch[:, 0, :, :] - inv_label_batch) * weight_batch
        self.diff[:, 1, :, :] = (
            loss_batch[:, 1, :, :] - label_batch) * weight_batch
        loss_batch[:, 0, :, :] = loss_batch[:, 0, :, :] * inv_label_batch
        loss_batch[:, 1, :, :] = loss_batch[:, 1, :, :] * label_batch
        # pdb.set_trace()

        loss_batch2 = loss_batch.sum(axis=1)
        log_loss_batch = log(loss_batch2)
        loss_matrix = log_loss_batch * weight_batch
        # print "valid_count : %d" % np.sum(weight_batch)
        # pdb.set_trace()
        top[0].data[...] = np.sum(loss_matrix) / np.sum(weight_batch)

    def backward(self, top, propagate_down, bottom):
        pdb.set_trace()
        for i in range(3):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / \
                np.sum(bottom[2].data[...])


if __name__ == "__main__":
    import ImageTransf as Transf
    import matplotlib.pylab as plt
    import os

    path = '/Users/naylorpeter/Documents/Histopathologie/ToAnnotate'
    out = "~/test/"

    crop = None
    crop = 4
    transform_list = [Transf.Identity(), Transf.Rotation(
        45, enlarge=False), Transf.Flip(1)]

    train = DataGen(path, crop, transform_list, split="train")
    test = DataGen(path, crop, transform_list, split="test")

    # Transf.ElasticDeformation(0,30,4)]#,
    # Transf.Rotation(45)]#,Transf.ElasticDeformation(0,30,4)]
