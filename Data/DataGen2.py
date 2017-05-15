import FIMM_histo.deconvolution as deconv
import glob
import numpy as np
from random import shuffle, randint, seed, sample
import os
from scipy import misc
import nibabel as ni
from Deprocessing.Morphology import generate_wsl
from UsefulFunctions.RandomUtils import CheckExistants, CheckFile
from UsefulFunctions.ImageTransf import Identity, flip_vertical, flip_horizontal
from UsefulFunctions.WeightMaps import ComputeWeightMap
import itertools
from skimage import measure
import pdb
from matplotlib.pylab as plt


class DataGen(object):

    def __init__(self, path, crop=1, size=None, transforms=None,
                 split="train", leave_out=1, seed_=None, name="optionnal",
                 img_format="RGB", wgt_param=None, Unet=False, 
                 return_path=False):

        self.path = path
        self.name = name
        self.transforms = transforms
        self.crop = crop
        self.split = split
        self.leave_out = leave_out
        self.seed = seed_

        self.GetPatients(path)
        self.SortPatients()

        self.img_format = img_format

        
        self.Weight = False if wgt_param is None else True
        self.wgt_param = wgt_param
        self.return_path = return_path


        self.random_crop = True if size is not None else False
        self.size = size

        self.UNet_crop = Unet


    def ReLoad(self, split):
        self.split = split
        self.GetPatients(self.path)
        self.SortPatients()


    def __getitem__(self, key):


        if len(key) != 4:
            print "key given: ", key
            print "key length %d" % len(key)
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
        
        img_lbl_Mwgt = ()
        img_lbl_Mwgt += (self.LoadImage(img_path), )
        img_lbl_Mwgt += (self.LoadGT(lbl_path), )


        if self.Weight:
            ## add wegiht check for 0
            wgt_path = self.Weight_path(img_path)
            img_lbl_Mwgt += (self.LoadWeight(wgt_path), )

        f = self.transforms[key[2]]

        img_lbl_Mwgt = f._apply_(*img_lbl_Mwgt)  # change _apply_

        i = 0
        if self.crop != 1:
            for sub_el in self.DivideImage(*img_lbl_Mwgt):
                if i == key[3]:
                    img_lbl_Mwgt = sub_el
                    break
                else:
                    i += 1

        if self.random_crop:
            img_lbl_Mwgt = self.CropImgLbl(*img_lbl_Mwgt)
        
        if self.UNet_crop:
            img_lbl_Mwgt = self.Unet_cut(*img_lbl_Mwgt)

        if self.return_path:
            img_lbl_Mwgt += (img_path,)


        return img_lbl_Mwgt


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

        return image


    def LoadGT(self, path, normalize=True):

        image = ni.load(path)
        img = image.get_data()
        img = measure.label(img)
        wsl = generate_wsl(img[:,:,0])
        img[ img > 0 ] = 1
        wsl[ wsl > 0 ] = 1
        img[:,:,0] = img[:,:,0] - wsl
        if len(img.shape) == 3:
            img = img[:, :, 0].transpose()
        else:

            img = img.transpose()
        return img


    def LoadWeight(self, path):

        image = misc.imread(path)

        return image


    def Weight_path(self, img_path):

        to_substitute = "WEIGHTS/{}_{}_{}_{}".format(*self.wgt_param)
        wgt_path = img_path.replace("Slide", to_substitute)
        self.wgt_dir = os.path.join(self.path, to_substitute)

        w_0 = self.wgt_param[0]
        val = self.wgt_param[1:3]
        sigma = self.wgt_param[3]

        try:
            CheckExistants(self.wgt_dir)
        except:
            self.wgt_dir = ComputeWeightMap(self.path, w_0, val, sigma)

        return self.wgt_path


    def DivideImage(self, *iterable):

        n_img = len(iterable)
        x = iterable[0].shape[0]
        y = iterable[0].shape[1]

        num_per_side = int(np.sqrt(self.crop))

        x_step = x / num_per_side
        y_step = y / num_per_side
        i_old = 0

        for i in range(x_step, x + 1, x_step):
            j_old = 0
            for j in range(y_step, y + 1, y_step):
                res = ()
                for k in range(n_img):
                    res += (iterable[k][i_old:i, j_old:j],)
                j_old = j
                yield res

            i_old = i


    def CropImgLbl(self, *kargs):

        dim = kargs[0].shape
        x = dim[0]
        y = dim[1]
        x_prime = self.size[0]
        y_prime = self.size[1]
        x_rand = randint(0, x - x_prime)
        y_rand = randint(0, y - y_prime)
        res = ()
        for i in range(len(kargs)):
            res += (self.RandomCropGen(kargs[i], (x_prime, y_prime), (x_rand, y_rand)),)

        return res


    def RandomCropGen(self, img, size, shift):

        x_prime = size[0]
        y_prime = size[1]
        x_rand = shift[0]
        y_rand = shift[1]

        return img[x_rand:(x_rand + x_prime), y_rand:(y_rand + y_prime)]


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


    def RandomKey(self, rand):

        if not rand:

            return [0] * 4

        else:
            a = randint(0, len(self.patients_iter) - 1)
            numero = self.patients_iter[a]
            b = randint(0, len(self.patient_img[numero]) - 1)
            c = randint(0, len(self.transforms) - 1)
            d = randint(0, self.crop - 1)

            return [a, b, c, d]


    def NextKey(self, key):
        
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


    def SetPatient(self, num):

        if isinstance(num, list):
            test_patient = num
        else:
            test_patient = [num]

        train_patient = [el for el in self.patient_num if el not in test_patient]


        number_of_transforms = len(self.transforms)

        if self.split == "train":

            images_train = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(images_train) * self.crop * number_of_transforms
            self.patients_iter = train_patient

        else:

            images_test = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(images_test) * self.crop * number_of_transforms
            self.patients_iter = test_patient

        self.SetRandomList() ## needed? 


    def GeneratePossibleKeys(self):
        """
        Exhaustive list of all transformations
        """
        len_key = 4

        AllPossibleKeys = []
        i = 0
        
        for num in self.patients_iter:
            lists = ([i],)
            i += 1
            nber_per_patient = len(self.patient_img[num])
            lists += (range(nber_per_patient),)
            lists += (range(len(self.transforms)),)
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
        else:  
            self.key_iter += 1
        if self.key_iter == self.length:
            self.key_iter = 0

        return self.RandomList[self.key_iter]


    def GetPatients(self, path):

        folders = glob.glob(self.path + "/Slide_*")
        patient_num = []

        for el in folders:
            patient_num.append(el.split("_")[-1].split('.')[0])
        shuffle(patient_num)

        self.patient_num = patient_num
        self.patient_img = {el: glob.glob(self.path + "/Slide_{}".format(el) + "/*.png") for el in patient_num}


    def SortPatients(self):

        if self.seed is not None:
            seed(self.seed)

        n = len(self.patient_num)
        test_patient = sample(self.patient_num, self.leave_out)
        train_patient = [el for el in self.patient_num if el not in test_patient]
        number_of_transforms = len(self.transforms)

        if self.split == "train":

            train_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(train_images) * self.crop * number_of_transforms
            self.patients_iter = train_patient

        else:
            test_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(test_images) * self.crop * number_of_transforms
            self.patients_iter = test_patient


    def SetTransformation(self, list_object):

        self.transforms = list_object


    def SetPath(self, path):
        
        self.path = path
        self.GetPatients(path)
        self.SortPatients()



if __name__ == "__main__":

    path = 

    transf = [Identity, flip_vertical, flip_horizontal]
    transf_test = [Identity]

    size = 
    crop = 

    DG = DataGen(path, crop=crop, size=size, transforms=transf,
                 split="train", leave_out=1)
    DG_test = DataGen(path, crop=crop, size=size, transforms=transf_test,
                 split="test", leave_out=1)

