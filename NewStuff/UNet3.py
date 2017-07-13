# -*- coding: utf-8 -*-
### for objects
from UNetBatchNorm import UNetBatchNorm
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
### for main
from optparse import OptionParser
from DataGen3Class import DataGen3
from UsefulFunctions.ImageTransf import ListTransform
import pdb

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def print_dim(text ,tensor):
    print text, tensor.get_shape()
    print

class UNetMultiClass(UNetBatchNorm):

    def init_training_graph(self):
        with tf.name_scope('Evaluation'):
            logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(logits, axis=3)
            
            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(tf.cast(self.train_labels_node, tf.int32), squeeze_dims=[3]),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.train_labels_node, tf.int64), squeeze_dims=[3])
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('ClassPrediction'):
                flat_LabelInt = tf.reshape(LabelInt, [-1])
                flat_predictions = tf.reshape(self.predictions, [-1])
                self.cm = tf.confusion_matrix(flat_LabelInt, flat_predictions)
                flatten_confusion_matrix = tf.reshape(self.cm, [-1])
                total = tf.reduce_sum(self.cm)
                for i in range(self.NUM_LABELS):
                    name = "Label_{}".format(i)
                    TP = self.cm[i, i]
                    if i == 0:
                        TN = tf.reduce_sum(self.cm[1:, 1:])
                        FN = tf.reduce_sum(self.cm[1:, 0])
                        FP = tf.reduce_sum(self.cm[0, 1:])
                    elif i == 1:
                        TN = tf.add(tf.add(self.cm[0,0] , self.cm[0,2]), tf.add(self.cm[2,0], self.cm[2,2]))
                        FN = tf.add(self.cm[0,2], self.cm[1,2])
                        FN = tf.add(self.cm[0,1], self.cm[2,1])
                    elif i == 2:
                        TN = tf.reduce_sum(self.cm[:-1, :-1])
                        FN = tf.reduce_sum(self.cm[-1, :-1])
                        FP = tf.reduce_sum(self.cm[:-1, -1])

                    precision =  tf.divide(TP, tf.add(TP, FP))
                    recall = tf.divide(TP, tf.add(TP, FN))
                    num = tf.multiply(precision, recall)
                    dem = tf.add(precision, recall)
                    F1 = tf.scalar_mul(2, tf.divide(num, dem))
                    Nprecision = tf.divide(TN, tf.add(TN, FN))
                    MeanAcc = tf.divide(tf.add(precision, Nprecision) ,2)

                    tf.summary.scalar(name + '_Precision', precision)
                    tf.summary.scalar(name + '_Recall', recall)
                    tf.summary.scalar(name + '_F1', F1)
                    tf.summary.scalar(name + '_Performance', MeanAcc)


            self.train_prediction = tf.nn.softmax(logits)

            self.test_prediction = tf.nn.softmax(logits)

        tf.global_variables_initializer().run()

        print('Computational graph initialised')
    def Validation(self, DG_TEST, step):
        n_test = DG_TEST.length
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 


        l, acc = 0., 0.
        cm = np.zeros((3,3))


        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l_tmp, acc_tmp, cm_tmp = self.sess.run([self.loss, self.accuracy, self.cm], feed_dict=feed_dict)
            l += l_tmp
            acc += acc_tmp
            cm += cm_tmp

        l, acc = np.array([l, acc]) / n_batch

        summary = tf.Summary()
        summary.value.add(tag="Test/Accuracy", simple_value=acc)
        summary.value.add(tag="Test/Loss", simple_value=l)
        
        for i in range(self.NUM_LABELS):
            name = "Label_{}".format(i)
            TP = cm[i, i]
            if i == 0:
                TN = cm[1:, 1:].sum()
                FN = cm[1:, 0].sum()
                FP = cm[0, 1:].sum()
            elif i == 1:
                TN = cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]
                FN = cm[0,2] + cm[1,2]
                FN = cm[0,1] + cm[2,1]
            elif i == 2:
                TN = cm[:-1, :-1].sum()
                FN = cm[-1, :-1].sum()
                FP = cm[:-1, -1].sum()
            precision =  float(TP) / (TP + FP)
            recall = float(TP) / (TP + FN)
            num = precision * recall
            dem = precision + recall
            F1 = 2 * (num / dem)
            Nprecision = float(TN) / (TN + FN)
            MeanAcc = (precision + Nprecision) / 2
            summary.value.add(tag="Test/" + name + "_F1", simple_value=F1)
            summary.value.add(tag="Test/" + name + "_Recall", simple_value=recall)
            summary.value.add(tag="Test/" + name + "_Precision", simple_value=precision)
            summary.value.add(tag="Test/" + name + "_Performance", simple_value=MeanAcc)

        self.summary_test_writer.add_summary(summary, step)

        print('  Validation loss: %.1f' % l)
        print('       Accuracy: %1.f%% \n   ' % (acc * 100))
        print('  Confusion matrix: \n')
        lb = ["Background", "Nuclei", "NucleiBorder"]
        print_cm(cm, lb)
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)


    def error_rate(self, predictions, labels, iter):
        predictions = np.argmax(predictions, 3)
        labels = labels[:,:,:,0]

        cm = confusion_matrix(labels.flatten(), predictions.flatten(), labels=[0, 1, 2]).astype(np.float)
        b, x, y = predictions.shape
        total = b * x * y
        acc = cm.diagonal().sum() / total
        error = 100 - acc

        return error, acc * 100, cm



    def train(self, DGTrain, DGTest, saver=True):
        epoch = DGTrain.length

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        tf.global_variables_initializer().run()

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        for step in range(steps):
            batch_data, batch_labels = DGTrain.Batch(0, self.BATCH_SIZE)
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
                error, acc, cm = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n ' % 
                      (l, acc))
                lb = ["Background", "Nuclei", "NucleiBorder"]
                print_cm(cm, lb)
                self.Validation(DGTest, step)


if __name__== "__main__":

    parser = OptionParser()
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
    parser.add_option("--weight_decay", dest="weight_decay", type="float",
                      help="weight decay value")
    (options, args) = parser.parse_args()

    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    LRSTEP = "10epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch
    PATH = options.path
    HEIGHT = 212
    WIDTH = 212
    N_TRAIN_SAVE = 2
    CROP = 4
    #pdb.set_trace()
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str


    transform_list, transform_list_test = ListTransform()

    DG_TRAIN = DataGen3(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file="mean_file.npy")

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    DG_TEST  = DataGen3(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file="mean_file.npy")
    DG_TEST.SetPatient(test_patient)

    model = UNetMultiClass(LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=HEIGHT,
                                       NUM_LABELS=3,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY)

    model.train(DG_TRAIN, DG_TEST)
