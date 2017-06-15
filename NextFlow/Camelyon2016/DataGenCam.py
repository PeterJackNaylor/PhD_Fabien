from os.path import join
from UsefulFunctions.ImageTransf import ListTransform
import numpy as np
import pandas as pd
from scipy.misc import imread
import pdb

class DataGen():
    def __init__(self, path,
                    transforms,
                    transforms_test,
                    size):
        self.path = path
        self.train = join(path, 'train.txt')
        self.test = join(path, 'test.txt')
        self.transforms = transforms
        self.transforms_test = transforms_test
        self.size = size
        self.key_train = 0
        self.key_test = 0
        self.LoadTables()

        self.p_train = len(transforms)
        self.p_test = len(transforms_test)

    def LoadTables(self):

        colname = ['x', 'y', 'size_x', 'size_y', 'ref_level',
                         'DomaineLabel', 'Label', 'Weight']
        self.table_train = pd.read_csv(self.train, sep = " ", header = None,
                            names=colname)
        self.table_test = pd.read_csv(self.test, sep = " ", header = None,
                            names=colname)

        self.n_train = self.table_train.shape[0]
        self.n_test = self.table_test.shape[0]



    def LoadItem(self, train):
        #pdb.set_trace()
        if train:
            l = self.table_train.loc[[self.key_train]]
        else:
            l = self.table_test.loc[[self.key_test]]

        para = [l["x"].values[0], l["y"].values[0], l["size_x"].values[0],
                l["size_y"].values[0], l["ref_level"].values[0]]
        domainLabel = l["DomaineLabel"].values[0]
        domainLabel = domainLabel.replace(".tif", "")
        Label = l["Label"].values[0]

        name = domainLabel + "_{}_{}_{}_{}_{}.png"
        img_path = join(self.path, name).format(*para)
        img = imread(img_path)[:,:,0:3]
        img = self.DataAugment(img, train)
        return img, Label

    def DataAugment(self, img, train):
        if train:
            p_trans = np.random.randint(0, self.p_train)
            t_img = self.transforms[p_trans]._apply_(img)
        else:
            p_trans = np.random.randint(0, self.p_test)
            t_img = self.transforms_test[p_trans]._apply_(img)
        return t_img[0]

    def AddKeyTrain(self, n):
        for i in range(n):
            self.key_train += 1
            if self.key_train == self.n_train:
                self.key_train = 0

    def AddKeyTest(self, n):
        for i in range(n):
            self.key_test += 1
            if self.key_test == self.n_test:
                self.key_test = 0        

    def NextItem(self, train):
        if train:
            img, lbl = self.LoadItem(train)
            self.AddKeyTrain(1)
        else:
            img, lbl = self.LoadItem(train)
            self.AddKeyTest(1)
        return img, lbl

    def NextBatch(self, train, bs):
        img_batch = np.zeros(shape=(bs, self.size[0], self.size[1], 3))
        lbl_batch = np.zeros(bs)
        for i in range(bs):
            if train:
                img_batch[i], lbl_batch[i] = self.LoadItem(train)
                self.AddKeyTrain(1)
            else:
                img_batch[i], lbl_batch[i] = self.LoadItem(train)
                self.AddKeyTest(1)
        return img_batch, lbl_batch        

if __name__ == "__main__":
    path = "/share/data40T_v2/CAMELYON16_precut"
    transforms, _ = ListTransform()
    size = (224, 224)

    DG = DataGen(path, transforms, _, size)

    train_batch, lbl_batch = DG.NextBatch(train= True, bs = 4)