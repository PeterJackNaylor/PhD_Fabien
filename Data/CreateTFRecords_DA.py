import pdb
import tensorflow as tf
from DataGenRandomT import DataGenRandomT
from DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from optparse import OptionParser
from UsefulFunctions.ImageTransf import Identity, Flip, OutOfFocus, ElasticDeformation, HE_Perturbation, HSV_Perturbation
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, MEAN_FILE, 
                   SEED, TEST_PATIENT, N_EPOCH, TYPE = "Normal",
                   SPLIT="train"):
    '''Not for UNet'''

    tfrecords_filename = OUTNAME
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    
    if TYPE == "Normal":
        DG = DataGenRandomT(PATH, split=SPLIT, crop=CROP, size=SIZE,
                        transforms=TRANSFORM_LIST, UNet=UNET,
                        mean_file=MEAN_FILE, seed_=SEED)

    elif TYPE == "3class":
        DG = DataGen3(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, 
                       mean_file=MEAN_FILE, seed_=SEED)
    elif TYPE == "ReducedClass":
        DG = DataGen3reduce(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, 
                       mean_file=MEAN_FILE, seed_=SEED)
    elif TYPE == "JUST_READ":
        DG = DataGenMulti(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, 
                       mean_file=MEAN_FILE, seed_=SEED)

    DG.SetPatient(TEST_PATIENT)
    N_ITER_MAX = N_EPOCH * DG.length


    original_images = []
    key = DG.RandomKey(False)
    if not UNET:
      for _ in range(N_ITER_MAX):
          key = DG.NextKeyRandList(0)
          img, annotation = DG[key]
  #        img = img.astype(np.uint8)
          annotation = annotation.astype(np.uint8)
          height = img.shape[0]
          width = img.shape[1]
      
          original_images.append((img, annotation))
          
          img_raw = img.tostring()
          annotation_raw = annotation.tostring()
          
          example = tf.train.Example(features=tf.train.Features(feature={
              'height': _int64_feature(height),
              'width': _int64_feature(width),
              'image_raw': _bytes_feature(img_raw),
              'mask_raw': _bytes_feature(annotation_raw)}))
          
          writer.write(example.SerializeToString())
    else:
      for _ in range(N_ITER_MAX):
          key = DG.NextKeyRandList(0)
          img, annotation = DG[key]
  #        img = img.astype(np.uint8)
          annotation = annotation.astype(np.uint8)
          height_img = img.shape[0]
          width_img = img.shape[1]

          height_mask = annotation.shape[0]
          width_mask = annotation.shape[1]
      
          original_images.append((img, annotation))
          
          img_raw = img.tostring()
          annotation_raw = annotation.tostring()
          
          example = tf.train.Example(features=tf.train.Features(feature={
              'height_img': _int64_feature(height_img),
              'width_img': _int64_feature(width_img),
              'height_mask': _int64_feature(height_mask),
              'width_mask': _int64_feature(width_mask),
              'image_raw': _bytes_feature(img_raw),
              'mask_raw': _bytes_feature(annotation_raw)}))
          
          writer.write(example.SerializeToString())


    writer.close()

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH,
                    BATCH_SIZE, N_THREADS, UNET):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    if not UNET:
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        height_img = tf.cast(features['height'], tf.int32)
        width_img = tf.cast(features['width'], tf.int32)

        height_mask = height_img
        width_mask = width_img

        const_IMG_HEIGHT = IMAGE_HEIGHT
        const_IMG_WIDTH = IMAGE_WIDTH

        const_MASK_HEIGHT = IMAGE_HEIGHT
        const_MASK_WIDTH = IMAGE_WIDTH


    else:
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height_img': tf.FixedLenFeature([], tf.int64),
            'width_img': tf.FixedLenFeature([], tf.int64),
            'height_mask': tf.FixedLenFeature([], tf.int64),
            'width_mask': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        height_img = tf.cast(features['height_img'], tf.int32)
        width_img = tf.cast(features['width_img'], tf.int32)

        height_mask = tf.cast(features['height_mask'], tf.int32)
        width_mask = tf.cast(features['width_mask'], tf.int32)

        const_IMG_HEIGHT = IMAGE_HEIGHT + 184
        const_IMG_WIDTH = IMAGE_WIDTH + 184

        const_MASK_HEIGHT = IMAGE_HEIGHT
        const_MASK_WIDTH = IMAGE_WIDTH


    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    
    image_shape = tf.stack([height_img, width_img, 3])
    annotation_shape = tf.stack([height_mask, width_mask, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    image_size_const = tf.constant((const_IMG_HEIGHT, const_IMG_WIDTH, 3), dtype=tf.int32)
    annotation_size_const = tf.constant((const_MASK_HEIGHT, const_MASK_WIDTH, 1), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    image_f = tf.cast(image, tf.float32)
    annotation_f = tf.cast(annotation, tf.float32)
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image_f,
                                           target_height=const_IMG_HEIGHT,
                                           target_width=const_IMG_WIDTH)
    
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation_f,
                                           target_height=const_MASK_HEIGHT,
                                           target_width=const_MASK_WIDTH)

    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=10 + 3 * BATCH_SIZE,
                                                 num_threads=N_THREADS,
                                                 min_after_dequeue=10)
    
    return images, annotations




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

    OUTNAME = options.TFRecords
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
    

    CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, None, 
                   SEED, TEST_PATIENT, N_EPOCH,
                   TYPE=TYPE, SPLIT=SPLIT)
