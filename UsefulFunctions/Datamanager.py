import glob
import numpy as np
import random
from sklearn.cross_validation import KFold
from scipy import misc
import nibabel as ni
import pdb
from itertools import chain


class DataManager(object):

    def __init__(self, path, crop=None, name="optionnal"):

        self.path = path
        self.name = name
        self.transforms = None
        self.crop = crop

    def TrainingIteratorFold(self, fold):
        try:
            train_paths, test_paths = self.ValScheme[fold]
        except:
            self.prepare_sets()
            train_paths, test_paths = self.ValScheme[fold]
        if self.transforms is None:
            for path_nii in np.array(self.nii)[train_paths]:
                img, img_gt = self.LoadGTAndImage(path_nii)
                if Normal:
                    yield img, img_gt, "Id"
                else:
                    i = 0
                    for subimg, subimg_gt in zip(self.DivideImage(img), self.DivideImage(img_gt)):
                        yield subimg, subimg_gt, "Id_sub_{}".format(i)
                        i += 1
        else:
            for path_nii in np.array(self.nii)[train_paths]:
                img, img_gt = self.LoadGTAndImage(path_nii)
                if Normal:
                    for f in self.transforms:
                        yield f._apply_(img), f._apply_(img_gt), f.name
                else:
                    for f in self.transforms:
                        i = 0
                        for subimg, subimg_gt in zip(self.DivideImage(f._apply_(img)), self.DivideImage(f._apply_(img_gt))):
                            yield subimg, subimg_gt, f.name + "_{}".format(i)
                            i += 1

    def TrainingIteratorLeaveValOut(self):
        if self.crop is None:
            Normal = True
        else:
            Normal = False
        fold = 0
        try:
            train_paths, test_paths = self.ValScheme[fold]
        except:
            self.prepare_sets()
            train_paths, test_paths = self.ValScheme[fold]
        all_paths = np.concatenate([train_paths, test_paths])
        if self.transforms is None:
            for path_nii in np.array(self.nii)[all_paths]:
                img, img_gt = self.LoadGTAndImage(path_nii)
                if Normal:
                    yield img, img_gt, "Id"
                else:
                    i = 0
                    for subimg, subimg_gt in self.CropIterator(img, img_gt):
                        yield subimg, subimg_gt, "Id_sub_{}".format(i)
                        i += 1
        else:
            for path_nii in np.array(self.nii)[all_paths]:
                img, img_gt = self.LoadGTAndImage(path_nii)
                if Normal:
                    for f in self.transforms:
                        yield f._apply_(img), f._apply_(img_gt), f.name
                else:
                    for f in self.transforms:
                        i = 0
                        for subimg, subimg_gt in zip(self.DivideImage(f._apply_(img)),
                                                     self.DivideImage(f._apply_(img_gt))):
                            yield subimg, subimg_gt, f.name + "_{}".format(i)
                            i += 1

    def ValIterator(self):
        for path_nii in np.array(self.nii)[np.array(self.ValScheme['score_set'])]:
            img, img_gt = self.LoadGTAndImage(path_nii)
            if self.crop is None:
                yield img, img_gt, "Id"
            else:
                i = 0
                for subimg, subimg_gt in zip(self.DivideImage(img), self.DivideImage(img_gt)):
                    yield subimg, subimg_gt, "Id_sub_{}".format(i)
                    i += 1

    def get_files(self, path):
        # Getting all nii.gz and png files in a 2 fold directory
        folders = glob.glob(path + "/*")
        all_nii = []
        all_png = []
        for fold in folders:
            temp_nii = glob.glob(fold + "/*.nii.gz")
            temp_png = glob.glob(fold + "/*.png")
            if len(temp_nii) != 0:
                all_nii += temp_nii
            if len(temp_png) != 0:
                all_png += temp_png
        return all_png, all_nii

    def prepare_sets(self, leave_out=2, folds=3):
        dic = {}
        try:
            n = len(self.nii)
        except:
            self.png, self.nii = self.get_files(self.path)
            n = len(self.nii)

        dic['score_set'] = random.sample(range(n), leave_out)
        new_list = np.array([i for i in range(n) if i not in dic['score_set']])
        n_prime = len(new_list)
        kf = KFold(n_prime, n_folds=folds, shuffle=True, random_state=None)
        j = 0
        for train_index, test_index in kf:
            dic[j] = (new_list[train_index], new_list[test_index])
            j += 1
        self.ValScheme = dic

    def SetTransformation(self, list_object):
        self.transforms = list_object

    def LoadGTAndImage(self, path):
        # you give the ground truth path
        try:
            num = path.split('/')[-1].split('.')[0]
            path_image = [el for el in self.png if num ==
                          el.split('/')[-1].split('.')[0]][0]
        except:
            self.png, self.nii = self.get_files(self.path)
            num = path.split('/')[-1].split('.')[0]
            path_image = [el for el in self.png if num ==
                          el.split('/')[-1].split('.')[0]][0]

        image = self.LoadImage(path_image)
        image_gt = self.LoadGT(path)
        return image, image_gt

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
        ImgImgIterator = chain(self.DivideImage(img), self.DivideImage(img_gt))
        return ImgImgIterator


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
# Useful plotting function

    def plot_comparison(original, modified, modification):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax1.set_adjustable('box-forced')
        ax2.imshow(modified, cmap=plt.cm.gray)
        ax2.set_title(modification)
        ax2.axis('off')
        ax2.set_adjustable('box-forced')

    path = '/home/naylor/Bureau/ToAnnotate'
    path = '/Users/naylorpeter/Documents/Python/ToAnnotate'
    out = "~/test/"

    crop = None
    crop = 4

    test = DataManager(path, crop)
    test.prepare_sets()

    # Transf.ElasticDeformation(0,30,4)]#,
    # Transf.Rotation(45)]#,Transf.ElasticDeformation(0,30,4)]
    transform_list = [Transf.Identity(), Transf.Rotation(
        45, enlarge=True), Transf.Flip(1)]
    test.SetTransformation(transform_list)
    i = 0
    for img, img_gt, name in test.TrainingIterator(fold=1):
        save_name = os.path.join
        im.save(save_name)
