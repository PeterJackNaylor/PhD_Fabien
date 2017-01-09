import random
import itertools
import pdb
from DataGen import DataGen
from WrittingTiff.tifffile import imread

from UsefulFunctions import ImageTransf as Transf
from UsefulFunctions.ImageTransf import Identity, flip_vertical, flip_horizontal
import matplotlib.pylab as plt
import numpy as np
import cPickle as pkl

def MakeDataGen(options):
    dgtrain = options.dgtrain
    dgtest = options.dgtest

    rawdata = options.rawdata
    rawdata_lbl = rawdata.replace('volume', 'labels')
    leaveout = options.leaveout
    seed = options.seed
    Unet = options.net == "UNet"

    transform_list = options.transform_list


    data_generator_train = DataGenIsbi2012(rawdata, rawdata_lbl,
                                  transform_list, split="train",
                                   leave_out=leaveout, seed=seed,
                                   Unet=Unet)
    pkl.dump(data_generator_train, open(dgtrain, "wb"))

    data_generator_test = DataGenIsbi2012(rawdata, rawdata_lbl,
                                  [Identity()], split="test",
                                  leave_out=leaveout, seed=seed,
                                  Unet=Unet)
    pkl.dump(data_generator_test, open(dgtest, "wb"))


def LoadSetImage(path):
    """This function is able to loads a multi page tif."""
    SetImage = imread(path)
    return SetImage


class DataGenIsbi2012(DataGen):
    def __init__(self, path, lbl_path, transforms, split ='train', leave_out=1, seed=None, Unet=True):


        self.path = path
        self.lbl_path = lbl_path
        self.transforms = transforms
        self.split = split
        self.leave_out = leave_out
        self.seed = seed
        self.UNet_crop = Unet
        self.n_f = len(transforms)
        self.SetDataSet()

        self.Weight = False
        self.WeightOnes = False
        self.return_path = False


    def LoadImage(self, val):
        return self.data_vol[val]

    def LoadLabel(self, val):
        return self.lbl_vol[val]

    def SetDataSet(self):
        random.seed(self.seed)
        data = LoadSetImage(self.path)
        label = LoadSetImage(self.lbl_path)
        num = data.shape[0]
        im_id = range(num)
        random.shuffle(im_id)
        if self.split == "train":
            self.data_vol = data[im_id[self.leave_out::]]
            self.lbl_vol = label[im_id[self.leave_out::]]
        elif self.split == "test":
            self.data_vol = data[im_id[0:self.leave_out]]
            self.lbl_vol = label[im_id[0:self.leave_out]]
        else:
            raise Exception('Not valid split name')
        self.number = self.data_vol.shape[0]
        self.length = self.number * self.n_f


    def __getitem__(self, key):

        val = key[0]
        transf = key[1]

        if len(key) != 2:
            raise Exception('key should be of 2 dimensions')
        if val >= self.number:
            raise Exception('Image index out of range (first dimension)')
        if transf >= self.n_f:
            raise Exception('Image index out of range (first dimension)')

        img = self.LoadImage(val)
        lbl = self.LoadLabel(val)

        img_lbl = (img, lbl)
#        pdb.set_trace()
        func = self.transforms[transf]
        img_lbl = func._apply_(*img_lbl)

        if self.UNet_crop:
            img_lbl = self.Unet_cut(*img_lbl)

        return img_lbl

    def Unet_cut(self, *kargs):
	# pdb.set_trace()
        dim = kargs[0].shape
        x = dim[0]
        y = dim[1]
        result = np.zeros(shape=(x+184, y+184))
        n = 92
        result[n:-n, n:-n] = kargs[0].copy()
        # top middle
        result[0:n, n:-n] = flip_horizontal(result[n:(2 * n), n:-n])
        # bottom middle
        result[-n::, n:-n] = flip_horizontal(result[-(2 * n):-n, n:-n])
        # left whole
        result[:, 0:n] = flip_vertical(result[:, n:(2 * n)])
        # right whole
        result[:, -n::] = flip_vertical(result[:, -(2 * n):-n])


        x_prime = 388
        y_prime = 388
        x_rand = random.randint(0, x - x_prime)
        y_rand = random.randint(0, y - y_prime)

        res = ()
        res += (self.RandomCropGen(result, (572, 572), (x_rand, y_rand)),)
        
        for i in range(1, len(kargs)):
            res += (self.RandomCropGen(kargs[i], (x_prime, y_prime), (x_rand, y_rand)),)
        return res



    def RandomKey(self, rand):
        if rand:
            val = random.randint(0, self.number - 1)
            transf = random.randint(0, self.n_f - 1)
            return (val, transf)
        else:
            return (0, 0)

    def NextKey(self, key):

        val = key[0]
        transf = key[1]

        transf += 1

        if transf == self.n_f:
            transf = 0
            val += 1

            if val == self.number:
                val = 0

        return (val, transf)

    def GeneratePossibleKeys(self):
        ran = (range(self.number), range(self.n_f))
        return list(itertools.product(*ran))

    def SetPatient(self, num):
        pass

if __name__ == "__main__":

    transform_list = [Transf.Identity(),
                      Transf.Flip(0),
                      Transf.Flip(1)]

    for rot in np.arange(320, 328, 4):
        transform_list.append(Transf.Rotation(rot, enlarge=True))

    for sig in [4]:
        transform_list.append(Transf.OutOfFocus(sig))

    for i in range(2):
        transform_list.append(Transf.ElasticDeformation(1.2, 24. / 512, 0.07))
    path = "/Users/naylorpeter/Downloads/isbi2012/train-volume.tif" 
    lbl_path = "/Users/naylorpeter/Downloads/isbi2012/train-labels.tif"
    print len(transform_list)
    datagen =  DataGenIsbi2012(path, lbl_path, transform_list, Unet=False)
    key = datagen.RandomKey(False)
    nu = 20
    fig, axes = plt.subplots(nu, 2, figsize=(10, 8 * nu))
    for i in range(nu):
        print key
        img, lbl = datagen[key]
        key = datagen.NextKey(key)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(lbl)
    plt.show()

