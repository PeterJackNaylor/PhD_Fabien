import tensorflow as tf
import numpy as np
from DataGen2 import DataGen, ListTransform
import os
from scipy.misc import imsave
import math
def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

def add_to_regularization(var):
    if var is not None:
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

class ConvolutionalNeuralNetwork:
    def __init__(
        self,
        LEARNING_RATE=0.01,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        STEPS=2000,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True):

        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.NUM_TEST = NUM_TEST
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LOG = LOG
        self.SEED = SEED

        self.sess = tf.InteractiveSession()

        self.sess.as_default()

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.DEBUG = DEBUG
        if self.DEBUG:
            self.WritteSummaryImages()


    def input_node_f(self):
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))

    def label_node_f(self):
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))

    def conv_layer_f(self, i_layer, w_var, strides, scope_name, padding="SAME"):
        with tf.name_scope(scope_name):
            return tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)

    def relu_layer_f(self, i_layer, biases, scope_name):
        with tf.name_scope(scope_name):
            return tf.nn.relu(tf.nn.bias_add(i_layer, biases))

    def weight_const_f(self, ks, inchannels, outchannels, stddev, scope_name, name="W"):
        with tf.name_scope(scope_name):
            return  tf.Variable(tf.truncated_normal([ks, ks, inchannels, outchannels],  # 5x5 filter, depth 32.
                              stddev=stddev,
                              seed=self.SEED))

    def weight_xavier_const_f(self, ks, inchannels, outchannels, scope_name, name="W"):
        xavier_std = math.sqrt( 1. / float(ks * ks * inchannels) )
        return self.weight_const_f(ks, inchannels, outchannels, xavier_std, scope_name, name=name)
    def biases_const_f(self, const, shape, scope_name, name="B"):
        with tf.name_scope(scope_name):
            return tf.Variable(tf.constant(const, shape=[shape]), name=name)

    def max_pool(self, i_layer, ksize=[1,2,2,1], strides=[1,2,2,1],
                 padding="SAME", name="MaxPool"):
        return tf.nn.max_pool(i_layer, ksize=ksize, strides=strides, 
                              padding=padding, name=name)

    def init_vars(self):
        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()

        self.conv1_weights = self.weight_const_f(5, self.NUM_CHANNELS, 8, 0.1, "conv1/")
        self.conv1_biases = self.biases_const_f(0.1, 8, "conv1/")

        self.conv2_weights = self.weight_const_f(5, 8, 8, 0.1, "conv2/")
        self.conv2_biases = self.biases_const_f(0.1, 8, "conv2/")

        self.conv3_weights = self.weight_const_f(5, 8, 8, 0.1, "conv3/")
        self.conv3_biases = self.biases_const_f(0.1, 8, "conv3/")

        self.logits_weight = self.weight_const_f(1, 8, 2, 0.1, "logits/")
        self.logits_biases = self.biases_const_f(0.1, 2, "logits/")

        self.keep_prob = tf.Variable(0.5)

        print('Model variables initialised')

    def WritteSummary(self):
        tf.summary.histogram('conv1_W', self.conv1_weights)
        tf.summary.histogram('conv2_W', self.conv2_weights)
        tf.summary.histogram('conv3_W', self.conv3_weights)
        tf.summary.histogram('conv1_B', self.conv1_biases)
        tf.summary.histogram('conv2_B', self.conv2_biases)
        tf.summary.histogram('conv3_B', self.conv3_biases)

    def WritteSummaryImages(self):
        tf.summary.image("Input", self.input_node, max_outputs=1)
        tf.summary.image("Label", self.train_labels_node, max_outputs=1)
        tf.summary.image("Pred", tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3), max_outputs=1)

    def init_model_architecture(self):

        self.conv1 = self.conv_layer_f(self.input_node, self.conv1_weights,
                                       [1,1,1,1], "conv1/")
        self.relu1 = self.relu_layer_f(self.conv1, self.conv1_biases, "conv1/")


        self.conv2 = self.conv_layer_f(self.relu1, self.conv2_weights,
                                       [1,1,1,1], "conv2/")
        self.relu2 = self.relu_layer_f(self.conv2, self.conv2_biases, "conv2/")

        self.conv3 = self.conv_layer_f(self.relu2, self.conv3_weights,
                                       [1,1,1,1], "conv3/")
        self.relu3 = self.relu_layer_f(self.conv3, self.conv3_biases, "conv3/")

        print('Model architecture initialised')

    def init_training_graph(self):
#        dropout = tf.nn.dropout(self.hidden1, self.keep_prob, seed=self.SEED)

        logits = self.conv_layer_f(self.relu3, self.logits_weight, [1,1,1,1], "logits/")
        #logits = self.relu_layer_f(logits_conv, self.logits_biases, "logits/")

        #one_hot_labels = tf.one_hot(tf.cast(self.train_labels_node, tf.uint8), depth = 2)
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #                           logits=logits, labels=one_hot_labels))
        self.predictions = tf.argmax(logits, axis=3)
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=tf.squeeze(tf.cast(self.train_labels_node, tf.int32), squeeze_dims=[3]),
                                                                      name="entropy")))
        #self.regularizers = (
        #    tf.nn.l2_loss(self.conv1_weights) + tf.nn.l2_loss(self.conv1_biases) +
        #    tf.nn.l2_loss(self.conv2_weights) + tf.nn.l2_loss(self.conv2_biases) +
        #    tf.nn.l2_loss(self.conv3_weights) + tf.nn.l2_loss(self.conv3_biases))

#        self.loss += 5e-4 * self.regularizers
        tf.summary.scalar("entropy", self.loss)
        self.batch = tf.Variable(0)

        self.train_prediction = tf.nn.softmax(logits)

        #predictions = tf.matmul(self.hidden1, self.fc2_weights) + self.fc2_biases
        #self.validation_prediction = tf.nn.softmax(predictions)
        self.test_prediction = tf.nn.softmax(logits)

        tf.global_variables_initializer().run()

        print('Computational graph initialised')


    def error_rate(self, predictions, labels, iter):

        predictions = np.argmax(predictions, 3)
        labels = labels[:,:,:,0]
        # for i in range(labels.shape[0]):
        #     imsave("/tmp/pred/pred_{}_{}.png".format(i, iter), predictions[i])
        #     imsave("/tmp/pred/label_{}_{}.png".format(i, iter), labels[i])

        correct = np.sum( predictions == labels )
        b, x, y = predictions.shape
        total = b * x * y
        print correct, total
        error = 100 - (100 * float(correct) / float(total))
        return error

    def optimization(self, var_list):
        #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(
        #    self.loss, global_step=self.batch)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list=var_list)
        if self.DEBUG:
            for grad, var in grads:
                add_gradient_summary(grad, var)
        self.optimizer = optimizer.apply_gradients(grads)        

    def train(self, DGTrain, DGTest):

        num_training = DGTrain.length

        self.learning_rate = tf.train.exponential_decay(
                             self.LEARNING_RATE,
                             self.batch * self.BATCH_SIZE,
                             num_training,
                             0.95,
                             staircase=True)

        trainable_var = tf.trainable_variables()
        if self.DEBUG:
            for var in trainable_var:
                add_to_regularization_and_summary(var)
        else:
            for var in trainable_var:
                add_to_regularization(var)

        self.optimization(trainable_var)

        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(self.LOG, graph=self.sess.graph)
        merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        Xval, Yval = DGTest.Batch(0, self.BATCH_SIZE)
        # for i in range(Xval.shape[0]):
        #     imsave("/tmp/image_{}.png".format(i), Xval[i])
        #     imsave("/tmp/label_{}.png".format(i), Yval[i,:,:,0])



        for step in range(steps):
            batch_data, batch_labels = DGTrain.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels}

            _, l, lr, predictions, s = self.sess.run(
                        [self.optimizer, self.loss, self.learning_rate,
                         self.train_prediction, merged_summary],
                        feed_dict=feed_dict)

            if step % self.N_PRINT == 0:
                summary_writer.add_summary(s, step)                
                error = self.error_rate(predictions, batch_labels, step)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % 
                      (l, error, lr))
                print('Validation error: %.1f%%' % self.error_rate(
                      self.test_prediction.eval(
                      feed_dict={self.input_node : Xval}),
                      (Yval).astype(np.float32), str(step) + "test"))



if __name__== "__main__":

    CUDA_NODE = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

    SAVE_DIR = "/tmp/object/230"
    N_ITER_MAX = 20000
    N_TRAIN_SAVE = 100
    N_TEST_SAVE = 100
    LEARNING_RATE = 0.001
    MEAN = np.array([122.67892, 116.66877 ,104.00699])
    HEIGHT = 224 
    WIDTH = 224
    CROP = 4
    PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
    PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
    BATCH_SIZE = 1
    LRSTEP = 200
    SUMMARY = True
    S = SUMMARY
    WEIGHT_FILE = "/home/pnaylor/Downloads/vgg16_weights.npz"


    transform_list, transform_list_test = ListTransform()
    DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=False)

    DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=False)

    model = ConvolutionalNeuralNetwork(LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=HEIGHT,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42)

    model.train(DG_TRAIN, DG_TEST)
