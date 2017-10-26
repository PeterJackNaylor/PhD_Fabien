from glob import glob
import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
import datetime

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_16s import FCN_16s

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

from optparse import OptionParser

if __name__ == '__main__':
    


    parser = OptionParser()

    parser.add_option("--checkpoint", dest="checkpoint", type="str",
                      help="Checkpoint to restore from")
    parser.add_option("--checksavedir", dest="checksavedir", type="str",
                      help="Directory of checkpoint to save to")
    parser.add_option("--tf_records", dest="tf_records", type="str",
                      help="tf_records to read from")

    parser.add_option("--log", dest="log", type="str",
                      help="Log directory")
    parser.add_option('--image_size', dest="size", type="int", 
                      help="Image size")
    parser.add_option('--labels', dest="labels", type="int", 
                      help="Number of labels")

    parser.add_option('--lr', dest="lr", type='float',  
                      help="Learning rate")
    parser.add_option('--n_print', dest="n_print", type='int', 
                      help="Prints and saves every n_print")
    parser.add_option('--iter', dest="iter", type="int", 
                      help="iter")

    (options, args) = parser.parse_args()


    slim = tf.contrib.slim

    tfrecord_filename = options.tf_records
    log_folder = options.log + "/log__fcn16__{}".format(options.lr)
    checksave = options.checksavedir + "/model__{}__fcn16s.ckpt".format(options.lr)


    slim = tf.contrib.slim

    image_train_size = [options.size, options.size]
    number_of_classes = options.labels



    class_labels = range(number_of_classes)
    class_labels.append(255)


    fcn_32s_checkpoint_path = glob(options.checkpoint + "/*.data*")[0].split(".data")[0] 

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=2)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    # Various data augmentation stages
    #image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

    # image = distort_randomly_image_color(image)

    resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


    resized_annotation = tf.squeeze(resized_annotation)

    image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=1,
                                                 capacity=3000,
                                                 num_threads=2,
                                                 min_after_dequeue=1000)

    upsampled_logits_batch, fcn_32s_variables_mapping = FCN_16s(image_batch_tensor=image_batch,
                                                               number_of_classes=number_of_classes,
                                                               is_training=True)


    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                         logits_batch_tensor=upsampled_logits_batch,
                                                                                        class_labels=class_labels)

    actual = tf.contrib.layers.flatten(tf.cast(annotation_batch, tf.int64))

    predicted_img = tf.argmax(upsampled_logits_batch, axis=3)

    predicted = tf.contrib.layers.flatten(predicted_img)
    #actual = tf.reshape(actual2, [1, 50176])
    tp = tf.count_nonzero(predicted * actual)
    tn = tf.count_nonzero((predicted - 1) * (actual - 1))
    fp = tf.count_nonzero(predicted * (actual - 1))
    fn = tf.count_nonzero((predicted - 1) * actual)
        
    # Calculate accuracy, precision, recall and F1 score.
    accuracy = tf.divide(tf.add(tp, tn), tf.add(tf.add(tp, fp), tf.add(fn, tn)))
    precision = tf.divide(tp, tf.add(tp, fp))
    recall = tf.divide(tp, tf.add(tp, fn))
    fmeasure = tf.divide(tf.multiply(tf.cast(2., tf.float64), tf.multiply(precision, recall)), tf.add(precision, recall))

    # Add metrics to TensorBoard.    
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Precision', precision)
    tf.summary.scalar('Recall', recall)
    tf.summary.scalar('f-measure', fmeasure)


    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                              labels=valid_labels_batch_tensor)

    #cross_entropy_sum = tf.reduce_sum(cross_entropies)

    cross_entropy_sum = tf.reduce_mean(cross_entropies)

    pred = tf.argmax(upsampled_logits_batch, dimension=3)

    probabilities = tf.nn.softmax(upsampled_logits_batch)

    with tf.name_scope('LearningRate'):
        decay_step = options.iter / 2
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
                                     options.lr,
                                     global_step,
                                     decay_step,
                                     0.96,
                                     staircase=True)

    with tf.variable_scope("adam_vars"):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_sum)


    #adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

    # Variable's initialization functions
    init_fn = slim.assign_from_checkpoint_fn(model_path=fcn_32s_checkpoint_path,
                                             var_list=fcn_32s_variables_mapping)

    global_vars_init_op = tf.global_variables_initializer()

    tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

    merged_summary_op = tf.summary.merge_all()

    summary_string_writer = tf.summary.FileWriter(log_folder)

    # Create the log folder if doesn't exist yet
    if not os.path.exists(log_folder):
         os.makedirs(log_folder)

    #optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)
        
    #The op for initializing the variables.
    local_vars_init_op = tf.local_variables_initializer()

    combined_op = tf.group(local_vars_init_op, global_vars_init_op)

    # We need this to save only model variables and omit
    # optimization-related and other variables.
    model_variables = slim.get_model_variables()
    saver = tf.train.Saver(model_variables)


    with tf.Session()  as sess:
        
        sess.run(combined_op)
        init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # Let's read off 3 batches just for example
        for i in xrange(0, options.iter):
            cross_entropy, F1, summary_string, _ = sess.run([ cross_entropy_sum, fmeasure,
                                                              merged_summary_op,
                                                              train_step ])
            
            if i % options.n_print == 0:
                print('\n Timestamp: {:%Y-%m-%d %H:%M:%S} :\n'.format(datetime.datetime.now()))
                print("Current loss: " + str(cross_entropy))
                print("Current F1: " + str(F1))
                summary_string_writer.add_summary(summary_string, i)
                save_path = saver.save(sess, checksave)
                print("Model saved in file: %s" % save_path)
                
                   
            
        coord.request_stop()
        coord.join(threads)
        
        save_path = saver.save(sess, checksave)
        print("Model saved in file: %s" % save_path)
        
    summary_string_writer.close()
