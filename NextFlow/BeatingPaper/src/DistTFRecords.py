from Data.CreateTFRecords import CreateTFRecord
from UsefulFunctions.ImageTransf import Identity, Flip, Rotation, OutOfFocus, ElasticDeformation, HE_Perturbation, HSV_Perturbation
import numpy as np
from optparse import OptionParser


def options_parser():

    parser = OptionParser()

    parser.add_option('--output', dest="TFRecords", type="string",
                      help="name for the output .tfrecords")
    parser.add_option('--path', dest="path", type="str",
                      help="Where to find the annotations")
    parser.add_option('--crop', dest="crop", type="int",
                      help="Number of crops to divide one image in")
    # parser.add_option('--UNet', dest="UNet", type="bool",
    #                   help="If image and annotations will have different shapes")
    parser.add_option('--size', dest="size", type="int",
                      help='first dimension for size')
    parser.add_option('--seed', dest="seed", type="int", default=42,
                      help='Seed to use, still not really implemented')  
    parser.add_option('--epoch', dest="epoch", type ="int",
                       help="Number of epochs to perform")  
    parser.add_option('--type', dest="type", type ="str",
                       help="Type for the datagen")  
    parser.add_option('--UNet', dest='UNet', action='store_true')
    parser.add_option('--no-UNet', dest='UNet', action='store_false')

    parser.add_option('--split', dest='split', type='str')
    parser.set_defaults(feature=True)

    (options, args) = parser.parse_args()
    options.SIZE = (options.size, options.size)
    return options

def ListTransform(n_rot=4, n_elastic=50, n_he=50, n_hsv = 50,
                  var_elast=[1.2, 24. / 512, 0.07], var_hsv=[0.01, 0.07],
                  var_he=[0.07, 0.07]):
    transform_list = [Identity(), Flip(0), Flip(1)]
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

    OUTNAME = options.TFRecords
    PATH = options.path
    CROP = options.crop
    SIZE = options.SIZE
    SPLIT = options.split
    var_elast = [1.3, 0.03, 0.15]
    var_he  = [0.01, 0.2]
    var_hsv = [0.2, 0.15]
    UNET = options.UNet
    SEED = options.seed
    N_EPOCH = options.epoch
    TYPE = options.type
    

    transform_list, transform_list_test = ListTransform(var_elast=var_elast,
                                                        var_hsv=var_hsv,
                                                        var_he=var_he) 
    if options.split == "train":
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach", "test"]
    elif options.split == "test":
        TEST_PATIENT = ["test"]
    elif options.split == "validation":
        options.split = "test"
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]
    TRANSFORM_LIST = transform_list



    CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, None, 
                   SEED, TEST_PATIENT, N_EPOCH,
                   TYPE=TYPE, SPLIT=SPLIT)
