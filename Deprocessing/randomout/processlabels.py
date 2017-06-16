from UsefulFunctions import ImageTransf as IT
import pdb
from Data.DataGen import DataGen
from Deprocessing.Morphology import generate_wsl
import skimage.measure as mea
from UsefulFunctions.RandomUtils import CheckOrCreate
from scipy.misc import imsave
import os

import matplotlib.pylab as plt

def plot_lbl(img):
    plt.imshow(img, cmap='jet')
    plt.show()


def plot(img):
    plt.imshow(img)
    plt.show()


def ImageLoop(path):
    datagen = DataGen(path, transforms=[
                      IT.Identity()], split="train", return_path=True)
    datagen.SetPatient("000000")
    key = datagen.RandomKey(True)
    for i in range(datagen.length):
        img, lbl, path = datagen[key]
        # pdb.set_trace()
        lbl = mea.label(lbl)
        lbl = lbl.astype('uint8')
        wsl = generate_wsl(lbl)
        lbl[lbl > 0] = 1
        wsl[wsl > 0] = 1
        lbl = lbl - wsl
        key = datagen.NextKey(key)
        path = path.replace("Slide", "GT_png")
        lbl[lbl > 0] = 255
        yield img, lbl, path


def num_ech(num):
    if num == "581910":
        return 1
    elif num == "141549":
        return 2
    elif num == "572123":
        return 3
    elif num == "498959":
        return 4
    elif num == "588626":
        return 5
    elif num == "162438":
        return 6
    elif num == "160120":
        return 7


def save_file(num, path):
    Slide_n = num_ech(num)
    folder_save = os.path.join(path, "Slide{}".format(Slide_n))
    CheckOrCreate(folder_save)
    num_image = len(os.listdir(folder_save))
    return os.path.join(folder_save, "Mask_{}.png".format(num_image + 1))

if __name__ == "__main__":
    outpath = "/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/masks"
    data = "/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/data"
    CheckOrCreate(data)
    CheckOrCreate(outpath)
    path = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate"
    for img, lbl, path in ImageLoop(path):
        save_folder = ["/"] + path.split('/')[0:-1]
        num = save_folder[-1].split('_')[-1]
        save_folder = save_file(num, outpath)
        save_img_folder = save_folder.replace(
            'masks', 'data').replace('Mask', 'Image')
        img__path = ["/"] + save_img_folder.split('/')[0:-1]
        CheckOrCreate(os.path.join(os.path.join(*img__path)))
        imsave(save_folder, lbl)
        imsave(save_img_folder, img)
        # CheckOrCreate(os.path.join(*save_folder))
        # imsave(path, lbl)
