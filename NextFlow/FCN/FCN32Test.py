
import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt
import pdb
from tf_image_segmentation.models.fcn_32s import FCN_32s

from matplotlib import pyplot as plt
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score
from Prediction.AJI import AJI_fast
from optparse import OptionParser
import pandas as pd
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

    (options, args) = parser.parse_args()


    restoremodel = options.checkpoint
    lr = restoremodel.split('__')[1]
    slim = tf.contrib.slim


    tfrecord_filename = options.tf_records

    number_of_classes = options.labels

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=1)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    # Fake batch for image and annotation by adding
    # leading empty axis.
    image_batch_tensor = tf.expand_dims(image, axis=0)
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

    # Be careful: after adaptation, network returns final labels
    # and not logits
    FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)


    pred, fcn_32s_variables_mapping = FCN_32s(image_batch_tensor=image_batch_tensor,
                                              number_of_classes=number_of_classes,
                                              is_training=False)

    # Take away the masked out values from evaluation
    weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255) )

    # Define the accuracy metric: Mean Intersection Over Union
    miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                       labels=annotation_batch_tensor,
                                                       num_classes=number_of_classes,
                                                       weights=weights)

    # The op for initializing the variables.
    initializer = tf.local_variables_initializer()

    saver = tf.train.Saver()

    ACC = 0
    AUC = 0
    AJI = 0
    F1 = 0
    RECALL = 0
    PRECISION = 0

    with tf.Session() as sess:
        
        sess.run(initializer)

        saver.restore(sess, restoremodel)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        
        for i in xrange(options.iter):
            
            image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, pred, update_op])
            y_true = annotation_np.flatten()
            y_pred = pred_np.flatten()
            ACC += accuracy_score(y_true, y_pred)
            AUC += auc(y_true, y_pred)
            AJI += AJI_fast(annotation_np[:,:,0], pred_np[0,:,:,0])
            F1 += f1_score(y_true, y_pred)
            PRECISION += precision_score(y_true, y_pred)
            RECALL += recall_score(y_true, y_pred)

            # Display the image and the segmentation result
            # upsampled_predictions = pred_np.squeeze()
            #plt.imshow(image_np)
            #plt.show()
            #visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut)
            
        coord.request_stop()
        coord.join(threads)
        
        res = sess.run(miou)
        ACC = ACC / options.iter
        AUC = AUC / options.iter
        F1 = F1 / options.iter
        PRECISION = PRECISION / options.iter
        RECALL = RECALL / options.iter
        results = {'ACC':[ACC,],'AUC':[AUC,], 'F1':[F1,], 
                    'PRECISION':[PRECISION,], 'RECALL':[RECALL,], 
                    "IU":[res,], "AJI":[AJI,]}
        df_results = pd.DataFrame(results)
         
        df_results.to_csv('fcn32__{}.csv'.format(lr))

