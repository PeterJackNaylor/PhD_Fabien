from ObjectOriented import ConvolutionalNeuralNetwork


from DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import os
import tensorflow as tf
from datetime import datetime
import skimage.io as io
import numpy as np
CREATE_TF = True

CUDA_NODE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

SAVE_DIR = "/tmp/object/newwwwww"
N_ITER_MAX = 2000
N_TRAIN_SAVE = 100
N_TEST_SAVE = 100
LEARNING_RATE = 0.001
SIZE = (224, 224)
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
PATH = "/data/users/pnaylor/Bureau/ToAnnotate"
PATH = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"
OUTNAME = 'firsttry.tfrecords'
SPLIT = 'train'
BATCH_SIZE = 2
LRSTEP = "10epoch"
SUMMARY = True
N_EPOCH = 1
SEED = 42
S = SUMMARY
TEST_PATIENT = ["141549", "162438"]
MEAN_FILE = None#"mean_file.npy"
UNET = False
transform_list, transform_list_test = ListTransform()
TRANSFORM_LIST = transform_list_test

DG = DataGenRandomT(PATH, split=SPLIT, crop=CROP, size=SIZE,
                        transforms=TRANSFORM_LIST, UNet=UNET,
                        mean_file=MEAN_FILE, seed_=SEED)
DG.SetPatient(TEST_PATIENT)
N_ITER_MAX = N_EPOCH * DG.length


original_images = []

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

for _ in range(3):
    key = DG.NextKeyRandList(0)
    img, annotation = DG[key]
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

writer.close()


reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    annotation_string = (example.features.feature['mask_raw']
                                .bytes_list
                                .value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    
    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))
    
    reconstructed_images.append((reconstructed_img, reconstructed_annotation))
    
