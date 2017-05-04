import numpy as np
import tensorflow as tf
from Data.DataGen import DataGen
from UsefulFunctions import ImageTransf as Transf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import pdb

LEARNING_RATE = 0.01

transform_list = [Transf.Identity(),
                  Transf.Flip(0),
                  Transf.Flip(1)]

width = 224
height = 224
dim = 3 
batch_size = 2
MEAN = np.array([104.00699, 116.66877, 122.67892])


dg = DataGen('/home/pnaylor/Documents/Data/ToAnnotate', crop = 4, 
             size=(width, height), transforms=transform_list)
dg_test = DataGen('/home/pnaylor/Documents/Data/ToAnnotate', split="test",
                  crop = 4, size=(width, height), transforms=transform_list)

def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features

        
        key = dg.NextKeyRandList(0)
        for batch_idx in range(0, dg.length, batch_size):
            images_batch = np.zeros(shape=(batch_size, width, height, dim))
            labels_batch = np.zeros(shape=(batch_size, width, height))
            for i in range(batch_size):

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
        for batch_idx in range(0, dg_test.length, batch_size):
            images_batch = np.zeros(shape=(batch_size, width, height, dim))
            labels_batch = np.zeros(shape=(batch_size, width, height))
            for i in range(batch_size):

                images_batch[i], labels_batch[i] = dg_test[key]
                images_batch[i] -= MEAN
                key = dg_test.NextKeyRandList(key)
            yield tf.convert_to_tensor(images_batch, dtype=tf.float32), tf.convert_to_tensor(labels_batch, dtype=tf.int32)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, width, height, dim])

    # Convolutional Layer #1

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)#,
        #kernel_initializer=tf.contrib.layers.xavier_initializer)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    logits = tf.layers.conv2d(
        inputs=conv3,
        filters=2,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)

    train_op = None
    loss = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

      # Generate Predictions
    predictions = {
          "classes": tf.argmax(
              input=logits, axis=3),
          "probabilities": tf.nn.softmax(
              logits, name="softmax_tensor"),
          "input": input_layer
    }

      # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
          mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(unused_argv):

    model_params = {"learning_rate": LEARNING_RATE}


    myclassifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/PangNet_model30")
    myclassifier.fit(input_fn = data_iterator().next, steps=20000)
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    eval_results = myclassifier.evaluate(
        input_fn = data_iterator_test().next, steps=100,  metrics=metrics)
    print(eval_results) 

import matplotlib.pylab as plt

def plot():
    myclassifier = learn.Estimator(                        
        model_fn=cnn_model_fn, model_dir="/tmp/PangNet_model30")
    output = list(myclassifier.predict(input_fn =  data_iterator_test().next))
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow((output[0]['input']+MEAN).astype('uint8'))
    axes[0,1].imshow(output[0]["probabilities"][:,:,0])
    axes[1,0].imshow((output[1]['input']+MEAN).astype('uint8'))
    axes[1,1].imshow(output[1]["probabilities"][:,:,0])
    plt.show()

if __name__ == "__main__":
    tf.app.run()
