import cPickle as pkl
from UsefulFunctions.usefulPloting import Contours
import caffe
import matplotlib.pylab as plt
from sys import maxint
import FIMM_histo.deconvolution as deconv
import pdb


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
        top[1].data[...] = self.label
        if self.datagen.Weight:
            top[2].data[...] = self.weight
        # pick next input
       # pdb.set_trace()
        self.Nextkey()

    def backward(self, top, propagate_down, bottom):
        pass

    def Nextkey(self):
        if self.random:
            self.key = self.datagen.RandomKey(True)
        else:
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


class DataGen(object):

    def __init__(self, path, crop=None, size=None, transforms=None,
                 split="train", leave_out=1, seed=None, name="optionnal",
                 img_format="RGB", Weight=False, WeightOnes=False):

        self.path = path
        self.name = name
        self.transforms = transforms
        self.crop = crop
        self.split = split
        self.leave_out = leave_out
        self.seed = seed
        self.get_patients(path, seed)
        self.Sort_patients()
        self.img_format = img_format
        self.Weight = Weight
        self.WeightOnes = WeightOnes
        if size is not None:
            self.random_crop = True
            self.size = size
        else:
            self.random_crop = False

    def ReLoad(self, split):
        self.split = split
        self.get_patients(self.path, self.seed)
        self.Sort_patients()

    def __getitem__(self, key):
        if self.transforms is None:
            len_key = 2
        elif self.crop == 1 or self.crop is None:
            len_key = 3
        else:
            len_key = 4

        if len(key) != len_key:
            print "key given: ", key
            print "key length %d" % len_key
            raise Exception('Wrong number of keys')

        if key[0] > len(self.patients_iter):
            raise Exception(
                "Value exceed number of patients available for {}ing.".format(self.split))
        numero = self.patients_iter[key[0]]
        n_patient = len(self.patient_img[numero])
        if key[1] > n_patient:
            raise Exception(
                "Patient {} doesn't have {} possible images.".format(self.patients_iter[key[0]], key[1]))
        if len_key > 2:
            if key[2] > len(self.transforms):
                raise Exception(
                    "Value exceed number of possible transformation for {}ing".format(self.split))
        if len_key == 4:
            if key[3] > self.crop - 1:
                raise Exception("Value exceed number of crops")

        img_path = self.patient_img[numero][key[1]]
        lbl_path = img_path.replace("Slide", "GT").replace(".png", ".nii.gz")

        img = self.LoadImage(img_path)
        lbl = self.LoadGT(lbl_path)

        i = 0
        if len_key == 4:
            for sub_image, sub_image_gt in self.CropIterator(img, lbl):
                if i == key[3]:
                    img = sub_image
                    lbl = sub_image_gt
                    break
                else:
                    i += 1
        if len_key > 2:
            f = self.transforms[key[2]]

            img = f._apply_(img)
            lbl = f._apply_(lbl)

        if self.Weight:
            wgt_path = img_path.replace("Slide", "WeightMap")
            weight = self.LoadWeight(wgt_path)

            i = 0
            if len_key == 4:
                for sub_weight in self.DivideImage(weight):
                    if i == key[3]:
                        weight = sub_weight
                        break
                    else:
                        i += 1
            if len_key > 2:
                f = self.transforms[key[2]]
                weight = f._apply_(weight)

        if self.random_crop:
            if not self.Weight:
                img, lbl = self.CropImgLbl(img, lbl, self.size)
            else:
                img, lbl, weight = self.CropImgLbl(
                    img, lbl, self.size, wgt=weight)
        if not self.Weight:
            return img, lbl
        else:
            if 0 in weight:
                weight[weight == 0] = 1
            # pdb.set_trace()
            return img, lbl, weight

    def get_patients(self, path, seed):
        # pdb.set_trace()
        if seed is not None:
            random.seed(seed)
        folders = glob.glob(path + "/Slide_*")
        patient_num = []
        for el in folders:
            patient_num.append(el.split("_")[-1].split('.')[0])
        random.shuffle(patient_num)

        self.patient_num = patient_num
        self.patient_img = {el: glob.glob(
            self.path + "/Slide_{}".format(el) + "/*.png") for el in patient_num}

        random.seed(random.randint(0, maxint - 1))

    def Sort_patients(self):
        n = len(self.patient_num)
        test_patient = random.sample(self.patient_num, self.leave_out)
        train_patient = [
            el for el in self.patient_num if el not in test_patient]
        if self.transforms is None:
            number_of_transforms = 1
        else:
            number_of_transforms = len(self.transforms)
        if self.split == "train":
            if self.crop is None:
                self.crop = 1
            self.length = np.sum([len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png"))
                                  for el in train_patient]) * self.crop * number_of_transforms

            self.patients_iter = train_patient
        else:
            if self.crop is None:
                self.crop = 1
            self.length = np.sum([len(glob.glob(
                self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]) * self.crop * number_of_transforms
            self.patients_iter = test_patient

    def SetTransformation(self, list_object):
        self.transforms = list_object

    def LoadGT(self, path, normalize=True):
        image = ni.load(path)
        img = image.get_data()
        new_img = np.zeros(shape=(img.shape[1], img.shape[0], 1))
        new_img[:, :, 0] = img[:, :, 0].transpose()
        new_img = new_img.astype("uint8")
        return new_img

    def LoadImage(self, path):
        if not hasattr(self, "img_format"):
            self.img_format = "RGB"
        image = misc.imread(path)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        if self.img_format == "HEDab":
            dec = deconv.Deconvolution()
            dec.params['image_type'] = 'HEDab'

            np_img = np.array(image)
            dec_img = dec.colorDeconv(np_img[:, :, :3])

            image = dec_img.astype('uint8')

        elif self.img_format == "HE":
            dec = deconv.Deconvolution()
            dec.params['image_type'] = 'HEDab'

            np_img = np.array(image)
            dec_img = dec.colorDeconv(np_img[:, :, :3])

            image = dec_img.astype('uint8')
            #image[:, :, 2] = image[:, :, 0]
        return image

    def LoadWeight(self, path):
        image = misc.imread(path)
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        if self.WeightOnes:
            image = np.ones_like(image)
        return image

    def DivideImage(self, img):
        if True:
            x = img.shape[0]
            y = img.shape[1]
            num_per_side = int(np.sqrt(self.crop))

            x_step = x / num_per_side
            y_step = y / num_per_side
            i_old = 0
            for i in range(x_step, x + 1, x_step):
                j_old = 0
                for j in range(y_step, y + 1, y_step):
                    sub_image = img[i_old:i, j_old:j]
                    j_old = j
                    # pdb.set_trace()
                    yield sub_image
                i_old = i

    def CropImgLbl(self, img, lbl, size, wgt=None, seed=None):
        if seed is not None:
            print "I set the seed here"
            random.seed(seed)
        dim = img.shape
        x = dim[0]
        y = dim[1]
        x_prime = size[0]
        y_prime = size[1]
        x_rand = random.randint(0, x - x_prime)
        y_rand = random.randint(0, y - y_prime)
        if wgt is None:
            return self.RandomCropGen(img, (x_prime, y_prime), (x_rand, y_rand)), self. RandomCropGen(lbl, (x_prime, y_prime), (x_rand, y_rand))
        else:
            return self.RandomCropGen(img, (x_prime, y_prime), (x_rand, y_rand)), self. RandomCropGen(lbl, (x_prime, y_prime), (x_rand, y_rand)), self.RandomCropGen(wgt, (x_prime, y_prime), (x_rand, y_rand))

    def RandomCropGen(self, img, size, shift):
        x_prime = size[0]
        y_prime = size[1]
        x_rand = shift[0]
        y_rand = shift[1]

        return img[x_rand:(x_rand + x_prime), y_rand:(y_rand + y_prime)]

    def CropIterator(self, img, img_gt):
        # pdb.set_trace()
        ImgImgIterator = zip(self.DivideImage(img), self.DivideImage(img_gt))
        return ImgImgIterator

    def RandomKey(self, rand):
        NoDim4 = self.crop is None or self.crop == 1
        NoDim3 = self.transforms is None
        if NoDim4:
            if NoDim3:
                dims = 2
            else:
                dims = 3
        else:
            dims = 4

        if not rand:
            return [0] * dims
        else:
            a = random.randint(0, len(self.patients_iter) - 1)
            numero = self.patients_iter[a]
            b = random.randint(0, len(self.patient_img[numero]) - 1)
            if not NoDim3:
                c = random.randint(0, len(self.transforms) - 1)
            if not NoDim4:
                d = random.randint(0, self.crop - 1)

            if not NoDim4:
                return [a, b, c, d]
            elif not NoDim3:
                return [a, b, c]
            else:
                return [a, b]

    def NextKey(self, key):
        if len(key) == 4:
            if key[3] == self.crop - 1:
                key[3] = 0  # crop
                if key[2] == len(self.transforms) - 1:
                    key[2] = 0  # transform list
                    numero = self.patients_iter[key[0]]
                    if key[1] == len(self.patient_img[numero]) - 1:
                        key[1] = 0
                        if key[0] == len(self.patients_iter) - 1:
                            key[0] = 0
                            return key
                        else:
                            key[0] += 1
                            return key
                    else:
                        key[1] += 1
                        return key
                else:
                    key[2] += 1
                    return key
            else:
                key[3] += 1
                return key
        elif len(key) == 3:
            if key[2] == len(self.transforms) - 1:
                key[2] = 0  # transform list
                numero = self.patients_iter[key[0]]
                if key[1] == len(self.patient_img[numero]) - 1:
                    key[1] = 0
                    if key[0] == len(self.patients_iter) - 1:
                        key[0] = 0
                        return key
                    else:
                        key[0] += 1
                        return key
                else:
                    key[1] += 1
                    return key
            else:
                key[2] += 1
                return key

        elif len(key) == 2:
            numero = self.patients_iter[key[0]]
            if key[1] == len(self.patient_img[numero]) - 1:
                key[1] = 0
                if key[0] == len(self.patients_iter) - 1:
                    key[0] = 0
                    return key
                else:
                    key[0] += 1
                    return key
            else:
                key[1] += 1
                return key
        else:
            raise Exception('Key is of wrong dimensions: {}'.format(len(key)))


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


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
        pdb.set_trace()
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
