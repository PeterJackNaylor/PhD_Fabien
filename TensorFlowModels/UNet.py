# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from Data.DataGen import DataGen
from UsefulFunctions import ImageTransf as Transf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import pdb

LEARNING_RATE = 0.01
DECAY_RATE= 0.95
BATCH_SIZE = 6
MODEL_DIR = "/share/data40T_v2/Peter/tmp_tensorflow/unet"

transform_list = [Transf.Identity(),
                  Transf.Flip(0),
                  Transf.Flip(1)]
for rot in np.arange(1, 360, 4):
    transform_list.append(Transf.Rotation(rot, enlarge=True))

for sig in [1, 2, 3, 4]:
    transform_list.append(Transf.OutOfFocus(sig))

for i in range(50):
    transform_list.append(Transf.ElasticDeformation(1.2, 24. / 512, 0.07))

perturbations = [ i / 100. for i in range(60, 140, 5)]
small_perturbation = [ i / 100. for i in range(80, 120, 5)]
for k_h in perturbations:
    for k_e in perturbations:
        transform_list.append(Transf.HE_Perturbation((k_h,0), (k_e,0), (1, 0)))
for k_b in perturbations:
    for k_s in small_perturbation:
        transform_list.append(Transf.HSV_Perturbation((1,0), (k_s,0), (k_b, 0))) 

<<<<<<< HEAD
width = 508
height = 508
dim = 3 
BATCH_SIZE = 2
MEAN = np.array([104.00699, 116.66877, 122.67892])
=======
transform_list_test = [Transf.Identity()]

width = 508
height = 508
dim = 3 
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb

MEAN = np.array([104.00699, 116.66877, 122.67892])

<<<<<<< HEAD
dg = DataGen('/data/users/pnaylor/Bureau/ToAnnotate', crop = 1, 
             size=(324, 324), transforms=transform_list, Unet=True)
dg_test = DataGen('/data/users/pnaylor/Bureau/ToAnnotate', split="test",
                  crop = 1, size=(324, 324), transforms=transform_list
=======
dg = DataGen('/share/data40T_v2/Peter/Data/ToAnnotate', crop = 1, 
             size=(324, 324), transforms=transform_list, Unet=True)
dg_test = DataGen('/share/data40T_v2/Peter/Data/ToAnnotate', split="test",
                  crop = 1, size=(324, 324), transforms=transform_list_test
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb
                  , Unet=True)
epoch = dg.length
print epoch

def Conv(input_layer, nout, ks, padding = "same"):
    return tf.layers.conv2d(
        inputs=input_layer,
        filters=nout,
        kernel_size=[ks, ks],
        padding=padding,
        activation=None)

def Maxpool(input_layer, ks = 2, stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(input_layer, 
                                pool_size=[ks,ks],
                                strides =[stride,stride],
                                padding=padding)

def ConvBnRelu(input_layer, nout, ks, padding = "same", name="relu"):
    conv = Conv(input_layer, nout, ks, padding)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)
    tf.summary.histogram(name, relu)
    return conv, bn, relu

def DeconvBnRelu(input_layer, nout, ks, padding = "VALID", name="relu"):
    deconv = tf.contrib.layers.conv2d_transpose(input_layer,
                                                nout,
                                                kernel_size = [ks, ks],
                                                stride = [2,2],
                                                padding = padding)
    bn = tf.layers.batch_normalization(deconv)
    relu = tf.nn.relu(bn)
    tf.summary.histogram(name, relu)
    return deconv, bn, relu



def Crop(input_layer, size):
    return tf.image.resize_image_with_crop_or_pad(input_layer, size, size)

def print_dim(text ,tensor):
    print text, tensor.get_shape()
    print 

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, width, height, dim])
    tf.summary.image("image" ,input_layer)
    # Convolutional Layer #1
<<<<<<< HEAD
    conv1_1, bn1_1, relu1_1 = ConvBnRelu(input_layer, 64, 3, "VALID") # n - 4
    conv1_2, bn1_2, relu1_2 = ConvBnRelu(relu1_1, 64, 3, "VALID") # n - 8
=======
    conv1_1, bn1_1, relu1_1 = ConvBnRelu(input_layer, 64, 3, "VALID", "relu1_1") # n - 4
    conv1_2, bn1_2, relu1_2 = ConvBnRelu(relu1_1, 64, 3, "VALID", "relu1_2") # n - 8
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb

    n1 = width - 4
    pool1 = Maxpool(relu1_2) # (n - 8) /2

<<<<<<< HEAD
    conv2_1, bn2_1, relu2_1 = ConvBnRelu(pool1, 128, 3, "VALID") # ((n-8)/2) - 4
    conv2_2, bn2_2, relu2_2 = ConvBnRelu(relu2_1, 128, 3, "VALID") # ((n-8)/2) - 8
    n2 = n1 / 2 - 4
    pool2 = Maxpool(relu2_2) # (((n-8)/2) - 8) /2

    conv3_1, bn3_1, relu3_1 = ConvBnRelu(pool2, 256, 3, "VALID") # (((n-8)/2) - 8) /2 - 4
    conv3_2, bn3_2, relu3_2 = ConvBnRelu(relu3_1, 256, 3, "VALID") # (((n-8)/2) - 8) /2 - 8
    n3 = n2 / 2 - 4
    pool3 = Maxpool(relu3_2) # ((((n-8)/2) - 8) /2 - 8 ) / 2

    conv4_1, bn4_1, relu4_1 = ConvBnRelu(pool3, 512, 3, "VALID") # ((((n-8)/2)-8)/2-8)/2 - 4
    conv4_2, bn4_2, relu4_2 = ConvBnRelu(relu4_1, 512, 3, "VALID") # ((((n-8)/2)-8)/2-8)/2 - 8
    n4 = n3 / 2 - 4
    pool4 = Maxpool(relu4_2) #(((((n-8)/2)-8)/2-8)/2 - 8)/2

    conv5_1, bn5_1, relu5_1 = ConvBnRelu(pool4, 1024, 3, "VALID") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 4
    conv5_2, bn5_2, relu5_2 = ConvBnRelu(relu5_1, 1024, 3, "VALID") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8
    n5 = n4 / 2 - 4

    deconv_To_4, bn_To_4, relu_To_4 = DeconvBnRelu(relu5_2, 512, 2, padding = "VALID")
=======
    conv2_1, bn2_1, relu2_1 = ConvBnRelu(pool1, 128, 3, "VALID", "relu2_1") # ((n-8)/2) - 4
    conv2_2, bn2_2, relu2_2 = ConvBnRelu(relu2_1, 128, 3, "VALID", "relu2_2") # ((n-8)/2) - 8
    n2 = n1 / 2 - 4
    pool2 = Maxpool(relu2_2) # (((n-8)/2) - 8) /2

    conv3_1, bn3_1, relu3_1 = ConvBnRelu(pool2, 256, 3, "VALID", "relu3_1") # (((n-8)/2) - 8) /2 - 4
    conv3_2, bn3_2, relu3_2 = ConvBnRelu(relu3_1, 256, 3, "VALID", "relu3_2") # (((n-8)/2) - 8) /2 - 8
    n3 = n2 / 2 - 4
    pool3 = Maxpool(relu3_2) # ((((n-8)/2) - 8) /2 - 8 ) / 2

    conv4_1, bn4_1, relu4_1 = ConvBnRelu(pool3, 512, 3, "VALID", "relu4_1") # ((((n-8)/2)-8)/2-8)/2 - 4
    conv4_2, bn4_2, relu4_2 = ConvBnRelu(relu4_1, 512, 3, "VALID", "relu4_2") # ((((n-8)/2)-8)/2-8)/2 - 8
    n4 = n3 / 2 - 4
    pool4 = Maxpool(relu4_2) #(((((n-8)/2)-8)/2-8)/2 - 8)/2

    conv5_1, bn5_1, relu5_1 = ConvBnRelu(pool4, 1024, 3, "VALID", "relu5_1") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 4
    conv5_2, bn5_2, relu5_2 = ConvBnRelu(relu5_1, 1024, 3, "VALID", "relu5_2") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8
    n5 = n4 / 2 - 4

    deconv_To_4, bn_To_4, relu_To_4 = DeconvBnRelu(relu5_2, 512, 2, "VALID", "derelu4")
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb
                                ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2

    n4_deconv = n5 * 2 
    diff4 = (n4 - n4_deconv) / 2
    
    crop_relu4_2 = tf.slice(relu4_2, [0, diff4, diff4, 0], [-1, n4_deconv, n4_deconv, 512])
<<<<<<< HEAD

    concat4 = tf.concat([relu_To_4, crop_relu4_2], axis=3)

    conv4_3, bn4_3, relu4_3 = ConvBnRelu(concat4, 512, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv4_4, bn4_4, relu4_4 = ConvBnRelu(relu4_3, 512, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_3, bn_To_3, relu_To_3 = DeconvBnRelu(relu4_4, 256, 2, padding = "VALID")
=======

    concat4 = tf.concat([relu_To_4, crop_relu4_2], axis=3)

    conv4_3, bn4_3, relu4_3 = ConvBnRelu(concat4, 512, 3, "VALID", "relu4_3") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv4_4, bn4_4, relu4_4 = ConvBnRelu(relu4_3, 512, 3, "VALID", "relu4_4") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_3, bn_To_3, relu_To_3 = DeconvBnRelu(relu4_4, 256, 2, "VALID", "derelu3")
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb

    n3_deconv = (n4_deconv - 4) * 2
    diff3 = (n3 - n3_deconv) / 2

    crop_relu3_2 = tf.slice(relu3_2, [0, diff3, diff3, 0], [-1, n3_deconv, n3_deconv, 256])
    concat3 = tf.concat([relu_To_3, crop_relu3_2], axis=3)

<<<<<<< HEAD
    conv3_3, bn3_3, relu3_3 = ConvBnRelu(concat3, 256, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv3_4, bn3_4, relu3_4 = ConvBnRelu(relu3_3, 256, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_2, bn_To_2, relu_To_2 = DeconvBnRelu(relu3_4, 128, 2, padding = "VALID")
=======
    conv3_3, bn3_3, relu3_3 = ConvBnRelu(concat3, 256, 3, "VALID", "relu3_3") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv3_4, bn3_4, relu3_4 = ConvBnRelu(relu3_3, 256, 3, "VALID", "relu3_4") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_2, bn_To_2, relu_To_2 = DeconvBnRelu(relu3_4, 128, 2, "VALID", "derelu2")
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb

    n2_deconv = (n3_deconv - 4) * 2
    diff2 = (n2 - n2_deconv) / 2 

    crop_relu2_2 = tf.slice(relu2_2, [0, diff2, diff2, 0], [-1, n2_deconv, n2_deconv, 128])
    concat2 = tf.concat([relu_To_2, crop_relu2_2], axis=3) 

    conv2_3, bn2_3, relu2_3 = ConvBnRelu(concat2, 128, 3, "VALID", "relu2_3") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv2_4, bn2_4, relu2_4 = ConvBnRelu(relu2_3, 128, 3, "VALID", "relu2_4") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_1, bn_To_1, relu_To_1 = DeconvBnRelu(relu2_4, 64, 2,  "VALID", "derelu1")


    n1_deconv = (n2_deconv - 4) * 2
    diff1 = (n1 - n1_deconv) / 2 
<<<<<<< HEAD

    crop_relu1_2 = tf.slice(relu1_2, [0, diff1, diff1, 0], [BATCH_SIZE, n1_deconv, n1_deconv, 64])
    concat1 = tf.concat([relu_To_1, crop_relu1_2], axis=3) 
    print_dim("concat1 :", concat1)
=======
>>>>>>> 2ef035bb0ae4e308c2ce5cad9bd66cc334ea92fb

    crop_relu1_2 = tf.slice(relu1_2, [0, diff1, diff1, 0], [BATCH_SIZE, n1_deconv, n1_deconv, 64])
    concat1 = tf.concat([relu_To_1, crop_relu1_2], axis=3) 

    conv1_3, bn1_3, relu1_3 = ConvBnRelu(concat1, 64, 3, "VALID", "relu1_3") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv1_4, bn1_4, relu1_4 = ConvBnRelu(relu1_3, 64, 3, "VALID", "relu1_4") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    
    logits = Conv(relu1_4, 2, 1, padding = "same")

    train_op = None
    loss = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        def mydecay(lr, global_step):
            return tf.train.exponential_decay(lr, 
                                              global_step,
                                              epoch / 2,
                                              DECAY_RATE)
        lr_decay = tf.train.exponential_decay(LEARNING_RATE, 
                                              tf.contrib.framework.get_global_step(),
                                              epoch / 2,
                                              DECAY_RATE,
                                              staircase=True)
        tf.summary.scalar("lr", lr_decay)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=lr_decay, #LEARNING_RATE,
            optimizer="SGD")
            #,
            #learning_rate_decay_fn=mydecay)


      # Generate Predictions
    predictions = {
          "classes": tf.argmax(
              input=logits, axis=3),
          "probabilities": tf.nn.softmax(
              logits, name="softmax_tensor"),
          "input": input_layer
#	  "label": labels
    }

      # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
          mode=mode, predictions=predictions, loss=loss, train_op=train_op)



def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        key = dg.NextKeyRandList(0)
        for batch_idx in range(0, dg.length, BATCH_SIZE):
            images_batch = np.zeros(shape=(BATCH_SIZE, width, height, dim))
            labels_batch = np.zeros(shape=(BATCH_SIZE, width - 184, height - 184))
            for i in range(BATCH_SIZE):

                images_batch[i], labels_batch[i] = dg[key]
                images_batch[i] -= MEAN
                key = dg.NextKeyRandList(key)
            yield tf.convert_to_tensor(images_batch, dtype=tf.float32), tf.convert_to_tensor(labels_batch, dtype=tf.int32)


def data_iterator_test():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features

        
        key = dg_test.NextKeyRandList(0)
        for batch_idx in range(0, dg_test.length, BATCH_SIZE):
            images_batch = np.zeros(shape=(BATCH_SIZE, width, height, dim))
            labels_batch = np.zeros(shape=(BATCH_SIZE, width - 184, height - 184))
            for i in range(BATCH_SIZE):

                images_batch[i], labels_batch[i] = dg_test[key]
                images_batch[i] -= MEAN
                key = dg_test.NextKeyRandList(key)
            yield tf.convert_to_tensor(images_batch, dtype=tf.float32), tf.convert_to_tensor(labels_batch, dtype=tf.int32)

def main(unused_argv):

    model_params = {"learning_rate": LEARNING_RATE}


    myclassifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/UNet")
    myclassifier.fit(input_fn = data_iterator().next, steps=20000)
    validation_metrics = {
    "accuracy":
        learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=learn.PredictionKey.
            CLASSES),
    "precision":
        learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=learn.PredictionKey.
            CLASSES),
    "recall":
        learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=learn.PredictionKey.
            CLASSES),
    "auc":
        learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_auc,
            prediction_key=learn.PredictionKey.
            CLASSES)
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=data_iterator_test().next,
        every_n_steps=epoch/16,
        eval_steps=dg_test.length,
        metrics=validation_metrics,
        early_stopping_metric='auc',
        early_stopping_metric_minimize=True,
        early_stopping_rounds=epoch/8)

    myclassifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir=MODEL_DIR,
        config=learn.RunConfig(save_checkpoints_secs=500))

    myclassifier.fit(
        input_fn = data_iterator().next, 
        steps=100 * epoch,
        monitors=[validation_monitor]
        )
        input_fn = data_iterator_test().next, steps=dg_test.length,  metrics=validation_metrics)
    print(eval_results)


import matplotlib.pylab as plt

def plot():
    myclassifier = learn.Estimator(                        
        model_fn=cnn_model_fn, model_dir=MODEL_DIR)
    output = list(myclassifier.predict(input_fn =  data_iterator_test().next))
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow((output[0]['input']+MEAN).astype('uint8')[92:-92,92:-92,:])
    axes[0,1].imshow(output[0]["probabilities"][:,:,0])
#    axes[0,2].imshow(output[0]["label"])
    axes[1,0].imshow((output[1]['input']+MEAN).astype('uint8')[92:-92,92:-92,:])
    axes[1,1].imshow(output[1]["probabilities"][:,:,0])
#    axes[1,2].imshow(output[1]["label"])
    plt.show()

if __name__ == "__main__":
    tf.app.run()
