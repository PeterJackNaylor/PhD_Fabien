import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from Data.DataGenRandomT import DataGenRandomT
from os.path import join
from UsefulFunctions.RandomUtils import CheckOrCreate
from optparse import OptionParser
from UsefulFunctions.ImageTransf import Identity, Flip, OutOfFocus, ElasticDeformation, HE_Perturbation, HSV_Perturbation
import numpy as np

def options_parser():

    parser = OptionParser()

    parser.add_option('--output', dest="output", type="string",
                      help="name for the output folder")
    parser.add_option('--path', dest="path", type="str",
                      help="Where to find the annotations")
    parser.add_option('--crop', dest="crop", type="int",
                      help="Number of crops to divide one image in")
    parser.add_option('--size', dest="size", type="int",
                      help='first dimension for size')
    parser.add_option('--seed', dest="seed", type="int", default=42,
                      help='Seed to use, still not really implemented')  
    parser.add_option('--epoch', dest="epoch", type ="int",
                       help="Number of epochs to perform")  
    parser.add_option('--elast1', dest="elast1", type ="float", default=0.,
                       help="First parameter to elast")
    parser.add_option('--elast2', dest="elast2", type ="float", default=0.,
                       help="Second parameter to elast")
    parser.add_option('--elast3', dest="elast3", type ="float", default=0.,
                       help="Third parameter to elast")
    parser.add_option('--he1', dest="he1", type ="float", default=0.,
                       help="First parameter to he1")
    parser.add_option('--he2', dest="he2", type ="float", default=0.,
                       help="Second parameter to he2")
    parser.add_option('--hsv1', dest="hsv1", type ="float", default=0.,
                       help="First parameter to hsv1")
    parser.add_option('--hsv2', dest="hsv2", type ="float", default=0.,
                       help="Second parameter to hsv2")
    parser.add_option('--type', dest="type", type ="str",
                       help="Type for the datagen")  
    parser.add_option('--UNet', dest='UNet', action='store_true')
    parser.add_option('--no-UNet', dest='UNet', action='store_false')

    parser.add_option('--train', dest='split', action='store_true')
    parser.add_option('--test', dest='split', action='store_false')
    parser.set_defaults(feature=True)

    (options, args) = parser.parse_args()
    options.SIZE = (options.size, options.size)
    return options

def ListTransform(n_rot=4, n_elastic=50, n_he=50, n_hsv = 50,
                  var_elast=[1.2, 24. / 512, 0.07], var_hsv=[0.01, 0.07],
                  var_he=[0.07, 0.07]):
    transform_list = [Identity(),
                      Flip(0),
                      Flip(1)]
    if n_rot != 0:
        for rot in np.arange(1, 360, n_rot):
            transform_list.append(Rotation(rot, enlarge=True))

    for sig in [1, 2, 3, 4]:
        transform_list.append(OutOfFocus(sig))

    for i in range(n_elastic):
        transform_list.append(ElasticDeformation(var_elast[0], var_elast[1], var_elast[2]))

    k_h = np.random.normal(1.,var_he[0], n_he)
    k_e = np.random.normal(1.,var_he[1], n_he)

    for i in range(n_he):
        transform_list.append(HE_Perturbation((k_h[i],0), (k_e[i],0), (1, 0)))


    k_s = np.random.normal(1.,var_hsv[0], n_hsv)
    k_v = np.random.normal(1.,var_hsv[1], n_hsv)

    for i in range(n_hsv):
        transform_list.append(HSV_Perturbation((1,0), (k_s[i],0), (k_v[i], 0))) 

    transform_list_test = [Identity()]

    return transform_list, transform_list_test

if __name__ == '__main__':

    options = options_parser()
    row = 4
    col = 4
    OUTNAME = options.output
    CheckOrCreate(OUTNAME)
    PATH = options.path
    CROP = options.crop
    SIZE = options.SIZE
    SPLIT = "train" if options.split else "test"

    n_elast = 100 if options.elast1 != 0 else 0
    n_hsv = 100 if options.hsv1 != 0 else 0
    n_he = 100 if options.he1 != 0 else 0

    transform_list, transform_list_test = ListTransform(0, n_elast, n_he, n_hsv,
                                                        var_elast=[options.elast1, options.elast2, options.elast3],
                                                        var_hsv=[options.hsv1, options.hsv2],
                                                        var_he=[options.he1, options.he2]) 
    TRANSFORM_LIST = transform_list
    UNET = options.UNet
    SEED = options.seed
    TEST_PATIENT = ["141549", "162438"]
    N_EPOCH = options.epoch
    TYPE = options.type
    
    DG = DataGenRandomT(PATH, split=SPLIT, crop=CROP, size=SIZE,
                        transforms=TRANSFORM_LIST, UNet=UNET,
                        mean_file=None, seed_=SEED)
    N_ITER_MAX = 1 #options.epoch * DG.length / (col * row)
    for _ in range(N_ITER_MAX):
        fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(6, 6))
        for i in range(row):
            for j in range(col):
                key = DG.NextKeyRandList(0)
                img, annotation = DG[key]
                ax[i, j].imshow(img[92:-92,92:-92])
                ax[i, j].axis('off')
        plt.savefig(join(OUTNAME, '{}.png'.format(_)), bbox_inches='tight')
