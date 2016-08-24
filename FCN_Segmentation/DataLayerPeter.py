import cPickle as pkl

import caffe


class DataLayerPeter(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - dir: path to image folder dir
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
        self.dir = params['dir']
        self.split = params['split']
        self.classifier_name = params['classifier_name']
        self.mean = np.array(params['mean'])
        self.crop = np.array(params['crop'])

        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        self.datagen = pkl.load(open(params['datagen'], 'rb'))
        self.datagen.ReLoad(self.split)
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
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
        self.data, self.label = self.datagen[self.key]
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.key = self.datagen.RandomKey(True)
        else:
            self.key = self.datagen.NextKey(self.key)

    def backward(self, top, propagate_down, bottom):
        pass

import glob
import numpy as np
import random
from sklearn.cross_validation import KFold
from scipy import misc
import nibabel as ni
import pdb
from itertools import chain


class DataGen(object):

    def __init__(self, path, crop=None, transforms=None, split="train", leave_out=1, seed=42, name="optionnal"):

        self.path = path
        self.name = name
        self.transforms = transforms
        self.crop = crop
        self.split = split
        self.leave_out = leave_out
        self.seed = seed
        self.get_patients(path, seed)
        self.Sort_patients()

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

        return img, lbl

    def get_patients(self, path, seed):
        # pdb.set_trace()
        random.seed(seed)
        folders = glob.glob(path + "/Slide_*")
        patient_num = []
        for el in folders:
            patient_num.append(el.split("_")[-1].split('.')[0])
        random.shuffle(patient_num)

        self.patient_num = patient_num
        self.patient_img = {el: glob.glob(self.path + "/Slide_{}".format(el) + "/*.png") for el in patient_num}

    def Sort_patients(self):
        n = len(self.patient_num)
        test_patient = random.sample(self.patient_num, self.leave_out)
        train_patient = [
            el for el in self.patient_num if el not in test_patient]

        if self.split == "train":
            if self.crop is None:
                self.crop = 1
            self.length = np.sum([len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png"))
                                  for el in train_patient]) * self.crop * len(self.transforms)

            self.patients_iter = train_patient
        else:
            if self.crop is None:
                self.crop = 1
            self.length = np.sum([len(glob.glob(
                self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]) * self.crop * len(self.transforms)
            self.patients_iter = test_patient

    def SetTransformation(self, list_object):
        self.transforms = list_object

    def LoadGT(self, path):
        image = ni.load(path)
        img = image.get_data()
        new_img = np.zeros(shape=(img.shape[1], img.shape[0], 1))
        new_img[:, :, 0] = img[:, :, 0].transpose()
        new_img = new_img.astype("uint8")
        return new_img

    def LoadImage(self, path):
        image = misc.imread(path)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
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
