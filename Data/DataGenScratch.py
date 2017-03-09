import random
import itertools
import pdb
from DataGen import DataGen
from WrittingTiff.tifffile import imread
import os
from UsefulFunctions import ImageTransf as Transf
from UsefulFunctions.ImageTransf import Identity, flip_vertical, flip_horizontal
import matplotlib.pylab as plt
import numpy as np
import cPickle as pkl
from RandomUtils import CheckExistants, CheckFile
from UsefulFunctions.UsefulOpenSlide import GetImage
import pandas as pd


class DataGenScratch(DataGen):
    """
    DataGen object that can deal with the Camelyon dataset
    """
    def __init__(self, CamelyonTextFilesFolder, split,  
                 transforms, size = None, random_crop = False, Unet = False,
                 img_format = "RGB", seed=1234, crop=1, name="CAM16",
                 pathfolder = "/share/data40T_v2/CAMELYON16_data"):
        self.path = CamelyonTextFilesFolder
        self.name = name
        self.transforms = transforms
        self.crop = crop
        self.split = split
        self.seed = seed
        self.img_format = img_format
        self.Weight = False
        self.WeightOnes = False
        self.return_path = False
        if size is not None:
            self.random_crop = True
            self.size = size
        else:
            self.random_crop = False
        self.UNet_crop = Unet
        self.pathfolder = pathfolder

        self.t_max = len(self.transforms)
        self.table, all_combi = self.LoadFiles()
        self.n_max = self.table.shape[0]
        self.BIG_N = 10000

    def __getitem__(self, key):
        n_row, n_trans = key
        img = self.LoadImage(n_row)
        lbl = self.LoadGT(n_trans)
        img_lbl_Mwgt = (img,)

        f = self.transforms[n_trans]
        img_lbl_Mwgt = f._apply_(*img_lbl_Mwgt)
        if self.random_crop:
            img_lbl_Mwgt = self.CropImgLbl(*img_lbl_Mwgt)
        if self.UNet_crop:
            img_lbl_Mwgt = self.Unet_cut(*img_lbl_Mwgt)

        img_lbl_Mwgt += (lbl, )
        return img_lbl_Mwgt


    def LoadFiles(self):
        colname = ['x', 'y', 'size_x', 'size_y', 'ref_level',
                         'DomaineLabel', 'Label', 'Weight']
        table = pd.read_csv(self.path, sep = " ", header = None,
                            names=colname)
        if self.split == "test":
            all_combi = list(itertools.product(table.index.values, [0]))
        else:
            all_combi = list(itertools.product(table.index.values, range(self.t_max)))
        random.shuffle(all_combi)
        df = pd.DataFrame(all_combi)
        df.to_csv("my_little_helper_{}.txt".format(self.split), header=None, sep=" ")
        return table, all_combi

    def ReLoad(self):
        self.table, all_combi = self.LoadFiles()

    def Weight_path(self):
        raise ValueError('Not possible with class DataGenScratch')
    def SetPatient(self, num):
        raise ValueError('Not possible with class DataGenScratch')
    def get_patients(self, path):
        raise ValueError('Not possible with class DataGenScratch')
    def Sort_patients(self):
        raise ValueError('Not possible with class DataGenScratch')
    def LoadWeight(self, path):
        raise ValueError('Not possible with class DataGenScratch')

    def LoadGT(self, index):
        l  = self.table.loc[[index]]
        Label = l["Label"].values[0]
        return Label

    def LoadImage(self, index):
        #pdb.set_trace()
        l  = self.table.loc[[index]]
        para = [l["x"].values[0], l["y"].values[0], l["size_x"].values[0],
                l["size_y"].values[0], l["ref_level"].values[0]]
        domainLabel = l["DomaineLabel"].values[0]
        state = domainLabel.split('_')[0]
        img_path = os.path.join(self.pathfolder, state, domainLabel)

        return np.array(GetImage(img_path, para))[:,:,0:3]

    def RandomKey(self, rand):
        if not rand:
            return [0] * 2
        else:
            a = random.randint(0, self.n_max - 1)
            b = random.randint(0, self.t_max - 1)
            return [a, b]
    def GeneratePossibleKeys(self):
        self.table, all_combi = self.LoadFiles()
        df = pd.DataFrame(all_combi)
        df.to_csv("my_little_helper_{}.txt".format(self.split), header=None, sep=" ")
        return df.head(n=self.BIG_N)

    def SetRandomList(self):
        RandomList = self.GeneratePossibleKeys()
        # random.shuffle(RandomList)
        self.RandomList = RandomList
        self.key_iter = 0

    def LoadBatch(self, n_min):
        if n_min + self.BIG_N > self.length:
            self.lines = self.length - (n_min + self.BIG_N)
        else:
            self.lines = self.BIG_N

        df = pd.read_csv("my_little_helper_{}.txt".format(self.split), header=None, sep=" ",
                        skiprows=n_min, nrows=self.lines)
        self.RandomList = df


    def NextKeyRandList(self, key):

        if not hasattr(self, "RandomList"):
            self.SetRandomList()
            self.key_iter = 0
            self.lines_seen = 0
            self.lines = self.BIG_N
            self.length = self.n_max * self.t_max
            self.lines_seen = 0
        else:  # pdb.set_trace()
            self.key_iter += 1

        if self.key_iter == self.lines:
            self.lines_seen += self.key_iter
            self.key_iter = 0
            if self.lines_seen == self.length:
                self.lines_seen = 0
                self.LoadBatch(0)
        return self.RandomList.loc[self.key_iter].values

    def NextKey(self, key):
        key[1] += 1
        if key[1] == self.t_max:
            key[1] = 0
            key[0] += 1
            if key[0] == self.n_max:
                key[0] = 0
        return key


    def SetPath(self, path):
        self.path = path

            
