import numpy as np
from DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import pdb
from os.path import join
from optparse import OptionParser
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage.morphology import distance_transform_cdt
from UsefulFunctions.RandomUtils import CheckOrCreate
from skimage.io import imsave
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk
def AddNoiseBinary(GT, p):
    x, y = GT.shape
    vec = GT.flatten()
    noise = np.random.binomial(1, p, len(vec))
    noisy_vec = 1 - ( vec * noise + (1 - vec) * (1 - noise) )
    GT_noise = noisy_vec.reshape(x,y)
    return GT_noise

def TruncatedNormal(loc, scale, size, min_v, max_v):
    vec = np.random.normal(loc=loc, scale=scale, size=size)
    def f(val):
        if min_v > val:
            return True
        elif val > max_v:
            return True
        else:
            return False
    res = np.array(map(f, vec))
    if True in res:
        n_size = res.sum()
        vec[res] = TruncatedNormal(loc, scale, n_size, min_v, max_v)
    return vec

def Noise(img, std):
    fl = img.flatten()
    vec = np.random.normal(loc=0, scale=std, size=len(fl))
    def g(val):
        if 0 > val:
            return -val
        else:
            return val
    noise = np.array(map(g, vec))
    res = fl + noise
    res = res.reshape(img.shape)
    res[res > 255.] = 255.
    return res.astype('uint8')

def AddNoise(GT, mu, std, std_2, min_v, max_v):
    GT_lbl = label(GT)
    x, y = GT.shape
    GT_noise = np.zeros(shape=(x, y, 3))
    for i in range(1, GT_lbl.max() + 1):

        col = TruncatedNormal(loc=mu, scale=std, size=3, min_v=min_v, max_v=max_v)
        #GT_noise[GT_lbl == i] = col.astype(int)
        nrow = GT_lbl[GT_lbl == i].shape[0]
        RANDOM = np.zeros(shape=(nrow, 3))
        for j in range(3):
            RANDOM[:, j] = TruncatedNormal(loc=col[j], scale=std_2, size=nrow, min_v=min_v, max_v=max_v)
        GT_noise[GT_lbl == i] = RANDOM

    return GT_noise.astype('uint8')
def DistanceWithoutNormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

def DistanceBinNormalise(bin_image):
    res = distance_transform_cdt(bin_image)
    props = regionprops(label(bin_image))
    res = res.astype('float')
    for p in props:
        MAX_VALUE = res[p.coords].max()
        for x, y in p.coords:
            res[x, y] = res[x, y] / float(MAX_VALUE)
    res = res * 255
    res = res.astype('uint8')
    return res

if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("--path", dest="path",type="string",
                      help="path to annotated dataset")
    parser.add_option("--output", dest="output",type="string",
                      help="out path")
    parser.add_option("-p", dest="p", type="float")
    parser.add_option("--test", dest="test", type="int")
    parser.add_option("--mu", dest="mu", type="int")
    parser.add_option("--sigma", dest="sigma", type="int")
    parser.add_option("--sigma2", dest="sigma2", type="int")
    (options, args) = parser.parse_args()


    path = "/data/users/pnaylor/Bureau/ToAnnotate"
    path = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"


    path = options.path
    transf, transf_test = ListTransform()

    size = (512, 512)
    crop = 1
    DG = DataGenRandomT(path, crop=crop, size=size, transforms=transf_test,
                 split="train", num="")
    Slide_train = join(options.output, "Slide_train")
    CheckOrCreate(Slide_train)
    Gt_train = join(options.output, "GT_train")
    CheckOrCreate(Gt_train)

    Slide_test = join(options.output, "Slide_test")
    CheckOrCreate(Slide_test)
    Gt_test = join(options.output, "GT_test")
    CheckOrCreate(Gt_test)

    for i in range(DG.length):
        key = DG.NextKeyRandList(0)
        img, gt = DG[key]  
        gt = label(gt)
        gt = dilation(gt, disk(3))
        GT_noise = AddNoise(gt, options.mu, options.sigma, options.sigma2, 1, 255)
        GT_noise = Noise(GT_noise, 5)
        if options.test > i:
            imsave(join(Slide_test, "test_{}.png").format(i), GT_noise)
            imsave(join(Gt_test, "test_{}.png").format(i), DistanceWithoutNormalise(gt))
        else:   
            imsave(join(Slide_train, "train_{}.png").format(i), GT_noise)
            imsave(join(Gt_train, "train_{}.png").format(i), DistanceWithoutNormalise(gt))
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(gt)
    ax[1].imshow(GT_noise)
    plt.show()
    #np.save(join(options.out, "mean_file.npy"), mean)
