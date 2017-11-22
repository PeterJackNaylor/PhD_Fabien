from glob import glob
from os.path import join
import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt
import pdb

from matplotlib import pyplot as plt
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, jaccard_similarity_score
from Prediction.AJI import AJI_fast
from optparse import OptionParser
import pandas as pd
from tf_image_segmentation.models.fcn_8s import FCN_8s
from UsefulFunctions.RandomUtils import CheckOrCreate, color_bin
from skimage.io import imsave


if __name__ == '__main__':
    


    parser = OptionParser()

    parser.add_option("--checkpoint", dest="checkpoint", type="str",
                      help="Checkpoint to restore from")
    parser.add_option("--tf_records", dest="tf_records", type="str",
                      help="tf_records to read from")

    parser.add_option('--labels', dest="labels", type="int", 
                      help="Number of labels")

    parser.add_option('--iter', dest="iter", type="int", 
                      help="iter")
    parser.add_option('--output', dest="output", type="str") 
    parser.add_option("--save_sample", dest="save_sample", type="str")

    (options, args) = parser.parse_args()
    CheckOrCreate(options.save_sample)
    save_folder = join(options.save_sample, options.tf_records.split('.')[0])
    CheckOrCreate(save_folder)


    restoremodel = ".".join(glob(join(options.checkpoint, '*_*'))[0].split('.')[:-1])
    slim = tf.contrib.slim


    tfrecord_filename = options.tf_records

    number_of_classes = options.labels

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=25)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    # Fake batch for image and annotation by adding
    # leading empty axis.
    image_batch_tensor = tf.expand_dims(image, axis=0)
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)
    # Be careful: after adaptation, network returns final labels
    # and not logits
    FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


    pred, fcn_16s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
                                              number_of_classes=number_of_classes,
                                              is_training=False)

    # Take away the masked out values from evaluation
    # weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255) )

    # Define the accuracy metric: Mean Intersection Over Union
    # miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
    #                                                   labels=annotation_batch_tensor,
    #                                                    num_classes=number_of_classes,
    #                                                   weights=weights)

    # The op for initializing the variables.
    initializer = tf.local_variables_initializer()

    saver = tf.train.Saver()

    ACC = []
    AUC = []
    AJI = []
    F1 = []
    RECALL = []
    PRECISION = []
    IU = []

    with tf.Session() as sess:
        
        sess.run(initializer)

        saver.restore(sess, restoremodel)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        
        for i in xrange(options.iter):
            
            image_np, annotation_np, pred_np = sess.run([image, annotation, pred])
            y_true = annotation_np.flatten()
            y_true[y_true > 0] = 1
            y_pred = pred_np.flatten()
            ACC.append(accuracy_score(y_true, y_pred))
            AUC.append(auc(y_true, y_pred))
            AJI.append(AJI_fast(annotation_np[:,:,0], pred_np[0,:,:,0]))
            F1.append(f1_score(y_true, y_pred, pos_label=1))
            PRECISION.append(precision_score(y_true, y_pred, pos_label=1))
            RECALL.append(recall_score(y_true, y_pred, pos_label=1))
            IU.append(jaccard_similarity_score(y_true, y_pred))

            j = 0
            xval_name = join(save_folder, "X_val_{}_{}.png".format(i, j))
            yval_name = join(save_folder, "Y_val_{}_{}.png".format(i, j))
            pred_bin_name = join(save_folder, "predbin_{}_{}.png".format(i, j))
            imsave(xval_name, image_np)
            imsave(yval_name, color_bin(annotation_np))
            imsave(pred_bin_name, color_bin(pred_np))
            
        coord.request_stop()
        coord.join(threads)
        
       # ACC = ACC / options.iter
       # AJI = AJI / options.iter
       # AUC = AUC / options.iter
       # F1 = F1 / options.iter
       # PRECISION = PRECISION / options.iter
       # RECALL = RECALL / options.iter
       # IU = IU / options.iter
        ORGAN = options.output.split('.')[0].split('_')[1]
        results = {'ORGAN':[ORGAN, ORGAN], 'NUMBER':list(range(options.iter)),'ACC':ACC,'AUC':AUC, 'F1':F1, 
                    'PRECISION':PRECISION, 'RECALL':RECALL, 
                    "IU":IU, "AJI":AJI}
        df_results = pd.DataFrame(results)
        df_results.to_csv(options.output)
