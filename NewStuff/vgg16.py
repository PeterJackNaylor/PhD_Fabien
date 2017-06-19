from UNetBatchNorm import UNetBatchNorm
import tensorflow as tf
from ObjectOriented import ConvolutionalNeuralNetwork
import tensorflow as tf
import os
import numpy as np
from UsefulFunctions.ImageTransf import ListTransform
import sys
sys.path.append('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/NextFlow/Camelyon2016/')
from DataGenCam import DataGen
import pdb
from datetime import datetime
import math 
from sklearn.metrics import confusion_matrix
from optparse import OptionParser

def print_dim(text ,tensor):
    print text, tensor.get_shape()
    print 

class VGG16(UNetBatchNorm):
    def __init__(
        self,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        N_EPOCHS=10,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        N_FEATURES = 16,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
#        self.NUM_TEST = NUM_TEST
        #self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED

        self.sess = tf.InteractiveSession()

        self.sess.as_default()

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.Saver()
        self.DEBUG = DEBUG
        self.N_EPOCHS = N_EPOCHS
        self.N_FEATURES = N_FEATURES


    def conv_layer_f(self, i_layer, w_var, scope_name, strides=[1,1,1,1], padding="SAME"):
        with tf.name_scope(scope_name):
            conv = tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)
            n_out = w_var.shape[3].value
            BN = self.BatchNorm(conv, n_out, self.is_training)
            #BN = conv
            #print_dim(scope_name, BN)
            return BN
    
    def init_vars(self):
        self.is_training = tf.placeholder(tf.bool)

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = N_FEATURES

        self.conv1_1weights = self.weight_xavier(3, self.NUM_CHANNELS, n_features, "conv1_1/")
        self.conv1_1biases = self.biases_const_f(0.1, n_features, "conv1_1/")

        self.conv1_2weights = self.weight_xavier(3, n_features, n_features, "conv1_2/")
        self.conv1_2biases = self.biases_const_f(0.1, n_features, "conv1_2/")



        self.conv2_1weights = self.weight_xavier(3, n_features, 2 * n_features, "conv2_1/")
        self.conv2_1biases = self.biases_const_f(0.1, 2 * n_features, "conv2_1/")

        self.conv2_2weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_2/")
        self.conv2_2biases = self.biases_const_f(0.1, 2 * n_features, "conv2_2/")



        self.conv3_1weights = self.weight_xavier(3, 2 * n_features, 4 * n_features, "conv3_1/")
        self.conv3_1biases = self.biases_const_f(0.1, 4 * n_features, "conv3_1/")

        self.conv3_2weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_2/")
        self.conv3_2biases = self.biases_const_f(0.1, 4 * n_features, "conv3_2/")

        self.conv3_3weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_3/")
        self.conv3_3biases = self.biases_const_f(0.1, 4 * n_features, "conv3_3/")



        self.conv4_1weights = self.weight_xavier(3, 4 * n_features, 8 * n_features, "conv4_1/")
        self.conv4_1biases = self.biases_const_f(0.1, 8 * n_features, "conv4_1/")

        self.conv4_2weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_2/")
        self.conv4_2biases = self.biases_const_f(0.1, 8 * n_features, "conv4_2/")

        self.conv4_3weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_3/")
        self.conv4_3biases = self.biases_const_f(0.1, 8 * n_features, "conv4_3/")



        self.conv5_1weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv5_1/")
        self.conv5_1biases = self.biases_const_f(0.1, 8 * n_features, "conv5_1/")

        self.conv5_2weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv5_2/")
        self.conv5_2biases = self.biases_const_f(0.1, 8 * n_features, "conv5_2/")

        self.conv5_3weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv5_3/")
        self.conv5_3biases = self.biases_const_f(0.1, 8 * n_features, "conv5_3/")



        self.fc6_weights = self.weight_xavier(7, 8 * n_features, 64 * n_features, "fc6/")
        self.fc6_biases = self.biases_const_f(0.1, 64 * n_features, "fc6/")

        self.fc7_weights = self.weight_xavier(1, 64 * n_features, 64 * n_features, "fc7/")
        self.fc7_biases = self.biases_const_f(0.1, 64 * n_features, "fc7/")


        self.logits_weights = self.weight_xavier(1, 64 * n_features, 2, "logits/")
        self.logits_biases = self.biases_const_f(0.1, 2, "logits/")

        self.keep_prob = tf.Variable(0.5, name="dropout_prob")

        print('Model variables initialised')

    def input_node_f(self):
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
    def label_node_f(self):
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE))




    def init_model_architecture(self):

        self.conv1_1 = self.conv_layer_f(self.input_node, self.conv1_1weights, "conv1_1/")
        self.relu1_1 = self.relu_layer_f(self.conv1_1, self.conv1_1biases, "conv1_1/")

        self.conv1_2 = self.conv_layer_f(self.relu1_1, self.conv1_2weights, "conv1_2/")
        self.relu1_2 = self.relu_layer_f(self.conv1_2, self.conv1_2biases, "conv1_2/")


        self.pool1_2 = self.max_pool(self.relu1_2, name="pool1_2")


        self.conv2_1 = self.conv_layer_f(self.pool1_2, self.conv2_1weights, "conv2_1/")
        self.relu2_1 = self.relu_layer_f(self.conv2_1, self.conv2_1biases, "conv2_1/")

        self.conv2_2 = self.conv_layer_f(self.relu2_1, self.conv2_2weights, "conv2_2/")
        self.relu2_2 = self.relu_layer_f(self.conv2_2, self.conv2_2biases, "conv2_2/")        


        self.pool2_3 = self.max_pool(self.relu2_2, name="pool2_3")


        self.conv3_1 = self.conv_layer_f(self.pool2_3, self.conv3_1weights, "conv3_1/")
        self.relu3_1 = self.relu_layer_f(self.conv3_1, self.conv3_1biases, "conv3_1/")

        self.conv3_2 = self.conv_layer_f(self.relu3_1, self.conv3_2weights, "conv3_2/")
        self.relu3_2 = self.relu_layer_f(self.conv3_2, self.conv3_2biases, "conv3_2/")     

        self.conv3_3 = self.conv_layer_f(self.relu3_2, self.conv3_3weights, "conv3_3/")
        self.relu3_3 = self.relu_layer_f(self.conv3_3, self.conv3_3biases, "conv3_3/")     


        self.pool3_4 = self.max_pool(self.relu3_2, name="pool3_4")


        self.conv4_1 = self.conv_layer_f(self.pool3_4, self.conv4_1weights, "conv4_1/")
        self.relu4_1 = self.relu_layer_f(self.conv4_1, self.conv4_1biases, "conv4_1/")

        self.conv4_2 = self.conv_layer_f(self.relu4_1, self.conv4_2weights, "conv4_2/")
        self.relu4_2 = self.relu_layer_f(self.conv4_2, self.conv4_2biases, "conv4_2/")

        self.conv4_3 = self.conv_layer_f(self.relu4_2, self.conv4_3weights, "conv4_3/")
        self.relu4_3 = self.relu_layer_f(self.conv4_3, self.conv4_3biases, "conv4_3/")


        self.pool4_5 = self.max_pool(self.relu4_2, name="pool4_5")


        self.conv5_1 = self.conv_layer_f(self.pool4_5, self.conv5_1weights, "conv5_1/")
        self.relu5_1 = self.relu_layer_f(self.conv5_1, self.conv5_1biases, "conv5_1/")

        self.conv5_2 = self.conv_layer_f(self.relu5_1, self.conv5_2weights, "conv5_2/")
        self.relu5_2 = self.relu_layer_f(self.conv5_2, self.conv5_2biases, "conv5_2/")

        self.conv5_3 = self.conv_layer_f(self.relu5_2, self.conv5_3weights, "conv5_3/")
        self.relu5_3 = self.relu_layer_f(self.conv5_3, self.conv5_3biases, "conv5_3/")


        self.pool5_fc = self.max_pool(self.relu5_3, name="pool5_fc")


        self.fc6 = self.conv_layer_f(self.pool5_fc, self.fc6_weights, "fc6/", padding="VALID")
        self.fc6_relu = self.relu_layer_f(self.fc6, self.fc6_biases, "fc6/")

        self.fc7 = self.conv_layer_f(self.fc6_relu, self.fc7_weights, "fc7/")
        self.fc7_relu = self.relu_layer_f(self.fc7, self.fc7_biases, "fc7/")



        self.conv_logit = self.conv_layer_f(self.fc7_relu, self.logits_weights, "logits/")
        self.relu_logit = self.relu_layer_f(self.conv_logit, self.logits_biases, "logits/")
        self.last = self.relu_logit

        print('Model architecture initialised')

    def init_training_graph(self):

        with tf.name_scope('Evaluation'):
            logits = self.last
            prob_b = tf.squeeze(logits, squeeze_dims=[1,2])
            self.predictions = tf.argmax(prob_b, axis=1)
            
            with tf.name_scope('Loss'):
                
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob_b,
                                                                          labels=tf.cast(self.train_labels_node, tf.int32),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.cast(self.train_labels_node, tf.int64)
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('Prediction'):

                self.TP = tf.count_nonzero(self.predictions * LabelInt)
                self.TN = tf.count_nonzero((self.predictions - 1) * (LabelInt - 1))
                self.FP = tf.count_nonzero(self.predictions * (LabelInt - 1))
                self.FN = tf.count_nonzero((self.predictions - 1) * LabelInt)

            with tf.name_scope('Precision'):

                self.precision = tf.divide(self.TP, tf.add(self.TP, self.FP))
                tf.summary.scalar('Precision', self.precision)

            with tf.name_scope('Recall'):

                self.recall = tf.divide(self.TP, tf.add(self.TP, self.TN))
                tf.summary.scalar('Recall', self.recall)

            with tf.name_scope('F1'):

                num = tf.multiply(self.precision, self.recall)
                dem = tf.add(self.precision, self.recall)
                self.F1 = tf.scalar_mul(2, tf.divide(num, dem))
                tf.summary.scalar('F1', self.F1)

            with tf.name_scope('MeanAccuracy'):
                
                Nprecision = tf.divide(self.TN, tf.add(self.TN, self.FN))
                self.MeanAcc = tf.divide(tf.add(self.precision, Nprecision) ,2)

            #self.batch = tf.Variable(0, name = "batch_iterator")

            self.train_prediction = tf.nn.softmax(logits)

            self.test_prediction = tf.nn.softmax(logits)

        tf.global_variables_initializer().run()

        print('Computational graph initialised')


    def Validation(self, DG, step):
        
        n_test = DG.n_test
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

        l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.


        for i in range(n_batch):
            Xval, Yval = DG.NextBatch(train=False, bs = self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp = self.sess.run([self.loss, self.accuracy, self.F1, self.recall, self.precision, self.MeanAcc], feed_dict=feed_dict)
            l += l_tmp
            acc += acc_tmp
            F1 += F1_tmp
            recall += recall_tmp
            precision += precision_tmp
            meanacc += meanacc_tmp

        l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch

        summary = tf.Summary()
        summary.value.add(tag="Test/Accuracy", simple_value=acc)
        summary.value.add(tag="Test/Loss", simple_value=l)
        summary.value.add(tag="Test/F1", simple_value=F1)
        summary.value.add(tag="Test/Recall", simple_value=recall)
        summary.value.add(tag="Test/Precision", simple_value=precision)
        summary.value.add(tag="Test/Performance", simple_value=meanacc)

        self.summary_test_writer.add_summary(summary, step)

        print('  Validation loss: %.1f' % l)
        print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)



    def train(self, DG, saver=True):

        epoch = DG.n_train 

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model(trainable_var)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        tf.global_variables_initializer().run()

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        merged_summary = tf.summary.merge_all()
        steps = self.N_EPOCHS * epoch

        for step in range(steps):
            batch_data, batch_labels = DG.NextBatch(train = True, bs = self.BATCH_SIZE)
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels,
                         self.is_training: True}

            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, merged_summary],
                        feed_dict=feed_dict)

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                      (l, acc, acc1, recall, prec, f1))
                self.Validation(DG, step)

    def WritteSummaryImages(self):
        tf.summary.image("Input", self.input_node, max_outputs=1)
        #tf.summary.image("Label", self.train_labels_node, max_outputs=1)
        #tf.summary.image("Pred", tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3), max_outputs=1)

    def error_rate(self, predictions, labels, iter):
        predictions = np.argmax(predictions, 3)

        cm = confusion_matrix(labels.flatten(), predictions.flatten(), labels=[0, 1]).astype(np.float)
        b, x, y = predictions.shape
        total = b * x * y

        TP = cm[1, 1]
        TN = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]

        acc = (TP + TN) / (TP + TN + FN + FP) * 100
        precision = TP / (TP + FP)
        acc1 = np.mean([precision, TN / (TN + FN)]) * 100
        recall = TP / (TP + FN)
        
        F1 = 2 * precision * recall / (recall + precision)
        error = 100 - acc

        return error, acc, acc1, recall * 100, precision * 100, F1 * 100


if __name__ == "__main__":

    parser = OptionParser()

#    parser.add_option("--gpu", dest="gpu", default="0", type="string",
#                      help="Input file (raw data)")

    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")

    parser.add_option("--log", dest="log",
                      help="log dir")

    parser.add_option("--learning_rate", dest="lr", type="float",
                      help="learning_rate")

    parser.add_option("--batch_size", dest="bs", type="int",
                      help="batch size")

    parser.add_option("--epoch", dest="epoch", type="int",
                      help="number of epochs")

    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")

    (options, args) = parser.parse_args()

#    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

    
    N_FEATURES = options.n_features
    N_TRAIN_SAVE = 100
    LEARNING_RATE = options.lr
    SAVE_DIR = options.log + "/" + "{}_{}" .format(N_FEATURES, LEARNING_RATE)
    
    MEAN = np.array([122.67892, 116.66877 ,104.00699])
    
    HEIGHT = 224 
    WIDTH = 224
    
    
    BATCH_SIZE = options.bs
    if N_FEATURES == 64:
        BATCH_SIZE = BATCH_SIZE // 2
    LRSTEP = "epoch/2"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch

    path = "/share/data40T_v2/CAMELYON16_precut"
    path = options.path
    transforms, _ = ListTransform()
    size = (HEIGHT, WIDTH)

    DG = DataGen(path, transforms, _, size)

#    train_batch, lbl_batch = DG.NextBatch(train= True, bs = 4)

    model = VGG16(LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=HEIGHT,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       N_EPOCHS=N_EPOCH,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       N_FEATURES=N_FEATURES,
                                       SEED=42)

    model.train(DG)