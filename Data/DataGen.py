import cPickle as pkl
import FIMM_histo.deconvolution as deconv
import pdb
import glob
import numpy as np
import random
from random import shuffle

from scipy import misc
import nibabel as ni
from UsefulFunctions.ImageTransf import Identity, flip_vertical, flip_horizontal
import copy
import itertools


def MakeDataGen(options):
    dgtrain = options.dgtrain
    dgtest = options.dgtest

    rawdata = options.rawdata
    leaveout = options.leaveout
    enlarge = options.enlarge
    Weight = options.Weight
    WeightOnes = options.WeightOnes
    loss = options.loss
    crop = options.crop
    crop_size = options.crop_size
    img_format = options.img_format
    seed = options.seed
    Unet = options.net == "UNet"

    transform_list = options.transform_list

    data_generator_train = DataGen(rawdata, crop=crop, size=crop_size,
                                   transforms=transform_list, split="train",
                                   leave_out=leaveout, seed=seed,
                                   img_format=img_format, Weight=Weight,
                                   WeightOnes=WeightOnes, Unet=Unet)
    pkl.dump(data_generator_train, open(dgtrain, "wb"))

    data_generator_test = DataGen(rawdata, crop=crop, size=crop_size,
                                  transforms=[Identity()], split="test",
                                  leave_out=leaveout, seed=seed,
                                  img_format=img_format, Weight=Weight,
                                  WeightOnes=WeightOnes, Unet=Unet)
    pkl.dump(data_generator_test, open(dgtest, "wb"))


class DataGen(object):

    def __init__(self, path, crop=None, size=None, transforms=None,
                 split="train", leave_out=1, seed=None, name="optionnal",
                 img_format="RGB", Weight=False, WeightOnes=False, Unet=False):

        self.path = path
        self.name = name
        self.transforms = transforms
        self.crop = crop
        self.split = split
        self.leave_out = leave_out
        self.seed = seed
        self.get_patients(path)
        self.Sort_patients()
        self.img_format = img_format
        self.Weight = Weight
        self.WeightOnes = WeightOnes
        if size is not None:
            self.random_crop = True
            self.size = size
        else:
            self.random_crop = False
        self.UNet_crop = Unet

    def ReLoad(self, split):
        self.split = split
        self.get_patients(self.path)
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

        if self.Weight:
            wgt_path = img_path.replace("Slide", "WeightMap")
            wgt = self.LoadWeight(wgt_path)
            img_lbl_Mwgt = (img, lbl, wgt)
        else:
            img_lbl_Mwgt = (img, lbl)

        i = 0
        if len_key == 4:
            for sub_el in self.DivideImage(*img_lbl_Mwgt):
                if i == key[3]:
                    img_lbl_Mwgt = sub_el
                    break
                else:
                    i += 1
        if len_key > 2:
            f = self.transforms[key[2]]
            img_lbl_Mwgt = f._apply_(*img_lbl_Mwgt)  # change _apply_

        if self.random_crop:

            img_lbl_Mwgt = self.CropImgLbl(*img_lbl_Mwgt)
        if self.Weight:
            if 0 in img_lbl_Mwgt[2]:
                img_lbl_Mwgt[2][img_lbl_Mwgt[2] == 0] = 1
        if not hasattr(self, "UNet_crop"):
            self.UNet_crop = False
        if self.UNet_crop:
            img_lbl_Mwgt = self.Unet_cut(*img_lbl_Mwgt)

        return img_lbl_Mwgt

    def SetPatient(self, num):
        # function for leave one out..
        test_patient = [num]
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
        self.SetRandomList()

    def get_patients(self, path):
        # pdb.set_trace()
        folders = glob.glob(path + "/Slide_*")
        patient_num = []
        for el in folders:
            patient_num.append(el.split("_")[-1].split('.')[0])
        shuffle(patient_num)

        self.patient_num = patient_num
        self.patient_img = {el: glob.glob(
            self.path + "/Slide_{}".format(el) + "/*.png") for el in patient_num}

    def Sort_patients(self):

        if self.seed is not None:
            random.seed(self.seed)

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
        try:
            new_img[:, :, 0] = img[:, :, 0].transpose()
        except:
            new_img[:, :, 0] = img[:, :].transpose()
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

    def DivideImage(self, *iterable):
        n_img = len(iterable)
        x = iterable[0].shape[0]
        y = iterable[0].shape[1]
        num_per_side = int(np.sqrt(self.crop))
        x_step = x / num_per_side
        y_step = y / num_per_side
        # print "step: ", x_step, y_step
        i_old = 0
        for i in range(x_step, x + 1, x_step):
            j_old = 0
            for j in range(y_step, y + 1, y_step):
                # pdb.set_trace()
                res = ()
                for k in range(n_img):
                    res += (iterable[k][i_old:i, j_old:j],)
                j_old = j
                yield res
            i_old = i

    def CropImgLbl(self, *kargs):
        size = self.size
        dim = kargs[0].shape
        x = dim[0]
        y = dim[1]
        x_prime = size[0]
        y_prime = size[1]
        x_rand = random.randint(0, x - x_prime)
        y_rand = random.randint(0, y - y_prime)
        res = ()
        for i in range(len(kargs)):
            res += (self.RandomCropGen(
                kargs[i], (x_prime, y_prime), (x_rand, y_rand)),)
        return res

    def Unet_cut(self, *kargs):
        dim = kargs[0].shape
        i = 0
        new_dim = ()
        for c in dim:
            if i < 2:
                ne = c + 184
            else:
                ne = c
            i += 1
            new_dim += (ne, )

        result = np.zeros(shape=new_dim)
        n = 92
        assert CheckNumberForUnet(
            dim[0] + 2 * n), "Dim not suited for UNet, it will create a wierd net"
        # middle
        result[n:-n, n:-n] = kargs[0].copy()
        # top middle
        result[0:n, n:-n] = flip_horizontal(result[n:(2 * n), n:-n])
        # bottom middle
        result[-n::, n:-n] = flip_horizontal(result[-(2 * n):-n, n:-n])
        # left whole
        result[:, 0:n] = flip_vertical(result[:, n:(2 * n)])
        # right whole
        result[:, -n::] = flip_vertical(result[:, -(2 * n):-n])

        res = (result, )
        for i in range(1, len(kargs)):
            res += (kargs[i],)
        return res

    def RandomCropGen(self, img, size, shift):
        x_prime = size[0]
        y_prime = size[1]
        x_rand = shift[0]
        y_rand = shift[1]

        return img[x_rand:(x_rand + x_prime), y_rand:(y_rand + y_prime)]

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

    def GeneratePossibleKeys(self):
        if self.transforms is None:
            len_key = 2
        elif self.crop == 1 or self.crop is None:
            len_key = 3
        else:
            len_key = 4
        AllPossibleKeys = []
        i = 0
        for num in self.patients_iter:
            lists = ([i],)
            i += 1
            nber_per_patient = len(self.patient_img[num])
            lists += (range(nber_per_patient),)
            if len_key > 2:
                lists += (range(len(self.transforms)),)
            if len_key > 3:
                lists += (range(self.crop),)
            AllPossibleKeys += list(itertools.product(*lists))

        return AllPossibleKeys

    def SetRandomList(self):
        RandomList = self.GeneratePossibleKeys()
        shuffle(RandomList)
        self.RandomList = RandomList
        self.key_iter = 0

    def NextKeyRandList(self, key):
        if not hasattr(self, "RandomList"):
            self.SetRandomList()
            self.key_iter = 0
        else:  # pdb.set_trace()
            self.key_iter += 1
        if self.key_iter == self.length:
            self.key_iter = 0

        return self.RandomList[self.key_iter]

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


def CheckNumberForUnet(n):
    ans = False
    if (n - 4) % 2 == 0:
        n = (n - 4) / 2
        if (n - 4) % 2 == 0:
            n = (n - 4) / 2
            if (n - 4) % 2 == 0:
                n = (n - 4) / 2
                if (n - 4) % 2 == 0:
                    n = (n - 4) / 2
                    ans = True

    return ans


if __name__ == "__main__":

    import sys
    #%matplotlib inline
    sys.path.append("/Users/naylorpeter/Documents/Python/PhD_Fabien")
    #from FCN_Segmentation.DataLayerPeter import DataGen
    from UsefulFunctions import ImageTransf as IT
    datagen = DataGen("/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/",
                      transforms=[IT.Identity()], Weight=True, Unet=True)
    datagen.ReLoad("train")
    key = datagen.RandomKey(True)
    import matplotlib.pylab as plt
    import skimage.morphology as skm
    nu = min(2, datagen.length)
    fig, axes = plt.subplots(nu, 3, figsize=(16, 180),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(nu):
        img, lbl, wgt = datagen[key]
        key = datagen.NextKey(key)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(wgt[:, :, 0])
        axes[i, 2].imshow(skm.label(lbl[:, :, 0]))
    plt.show()

"""    import glob
    import os
    from scipy import misc

    from scipy import ndimage
    val = (1, 3)

    folder = glob.glob(
        '/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/GT_*')

    for fold in folder:
        new_fold = fold.replace('GT', 'WeightMap')
        if not os.path.isdir(new_fold):
            os.mkdir(new_fold)
        list_nii = glob.glob(os.path.join(fold, '*.nii.gz'))
        for name in list_nii:
            # print name
            lbl = datagen.LoadGT(name)
            lbl_diff = skm.label(lbl[:, :, 0])
            res = WeightMap(lbl_diff, 10, val)
            new_name = name.replace('GT', 'WeightMap')
            new_name = new_name.replace('nii.gz', 'png')
            # print new_name
            misc.imsave(new_name, res)

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
"""