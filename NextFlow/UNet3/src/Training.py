from UNetBatchNorm_v2 import UNetBatchNorm
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime
from optparse import OptionParser
from UsefulFunctions.ImageTransf import ListTransform
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from Data.CreateTFRecords import read_and_decode
import pdb
from sklearn.metrics import confusion_matrix

lb_name = ['Background', 'Border', 'Cell']

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

def CM_DIV(cm, index, max_lab):
    list_FP = []
    list_TN = []
    list_FN = []
    for row in range(max_lab):
        for col in range(max_lab):
            if row != col:
                # making sure you are not on the diagonal
                element = cm[row, col]
                if row != index and col != index:
                    list_TN.append(element)
                elif row == index:
                    list_FN.append(element)
                elif col == index:
                    list_FP.append(element)
    return cm[index, index] ,list_TN, list_FP, list_FN

def GetCMInfo(cm, index, max_lab):

    TP, list_TN, list_FP, list_FN = CM_DIV(cm, index, max_lab)
    FP = np.sum(list_FP)
    FN = np.sum(list_FN)
    TN = np.sum(list_TN)
    return TP, TN, FP, FN

def GetCMInfo_TF(cm, index, max_lab):

    TP, list_TN, list_FP, list_FN = CM_DIV(cm, index, max_lab)
    FP = tf.reduce_sum(list_FP)
    FN = tf.reduce_sum(list_FN)
    TN = tf.reduce_sum(list_TN)
    return TP, TN, FP, FN

class UNet3(UNetBatchNorm):
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        STEPS=2000,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True,
        WEIGHT_DECAY=0.00005,
        LOSS_FUNC=tf.nn.l2_loss,
        N_FEATURES=16,
        N_EPOCH=1,
        N_THREADS=1,
        MEAN_FILE=None,
        DROPOUT=0.5,
        RESTORE="/tmp/net"):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.N_FEATURES = N_FEATURES
#        self.NUM_TEST = NUM_TEST
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED
        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        self.RESTORE = RESTORE
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, 3])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False

        self.sess = tf.InteractiveSession()

        self.sess.as_default()
        
        self.var_to_reg = []
        self.var_to_sum = []
        self.TF_RECORDS = TF_RECORDS
        self.init_queue(TF_RECORDS)

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.Saver()
        self.DEBUG = DEBUG
        self.loss_func = LOSS_FUNC
        self.weight_decay = WEIGHT_DECAY
    def Saver(self):
        print("Setting up Saver...")
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.RESTORE)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    def init_queue(self, tfrecords_filename):
        self.filename_queue = tf.train.string_input_producer(
                              [tfrecords_filename], num_epochs=10)
        with tf.device('/cpu:0'):
            self.image, self.annotation = read_and_decode(self.filename_queue, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS,
                                                          True,
                                                          self.NUM_CHANNELS)
            #self.annotation = tf.divide(self.annotation, 255.)
        print("Queue initialized")

    def init_training_graph(self):
        with tf.name_scope('Evaluation'):
            self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(self.logits, axis=3)
            
            with tf.name_scope('Loss'):
                LabelInt = tf.cast(self.train_labels_node, tf.int64)
                condition = tf.equal(LabelInt, 255)
                case_true = tf.multiply(tf.ones(LabelInt.get_shape(), dtype=tf.int64), 2)
                case_false = LabelInt
                LabelInt = tf.where(condition, case_true, case_false)

                condition = tf.equal(LabelInt, 127)
                case_true = tf.ones(LabelInt.get_shape(), dtype=tf.int64)
                case_false = LabelInt
                self.LabelInt = tf.where(condition, case_true, case_false)

                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(tf.cast(self.LabelInt, tf.int32), squeeze_dims=[3]),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.LabelInt, tf.int64), squeeze_dims=[3])

                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('ClassPrediction'):
                self.flat_LabelInt = tf.reshape(LabelInt, [-1])
                self.flat_predictions = tf.reshape(self.predictions, [-1])
                self.cm = tf.confusion_matrix(self.flat_LabelInt, self.flat_predictions, self.NUM_LABELS)
                flatten_confusion_matrix = tf.reshape(self.cm, [-1])
                total = tf.reduce_sum(self.cm)
                for i in range(self.NUM_LABELS):
                    name = "Label_{}".format(i)
                    TP, TN, FP, FN = GetCMInfo_TF(self.cm, i, self.NUM_LABELS)

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
                confusion_image = tf.reshape( tf.cast( self.cm, tf.float32),
                                            [1, self.NUM_LABELS, self.NUM_LABELS, 1])
                tf.summary.image('confusion', confusion_image)

            self.train_prediction = tf.nn.softmax(self.logits)

            self.test_prediction = self.train_prediction

        tf.global_variables_initializer().run()

        print('Computational graph initialised')

    def error_rate(self, predictions, labels, iter):
        predictions = np.argmax(predictions, 3)
        labels = labels[:,:,:,0]

        cm = confusion_matrix(labels.flatten(), predictions.flatten(), labels=range(self.NUM_LABELS)).astype(np.float)
        b, x, y = predictions.shape
        total = b * x * y
        acc = cm.diagonal().sum() / total
        error = 100 - acc

        return error, acc * 100, cm


    def Validation(self, DG_TEST, step):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc = 0., 0.
            cm = np.zeros((self.NUM_LABELS, self.NUM_LABELS))

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                #Yval = Yval / 255.
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, acc_tmp, cm_tmp, s = self.sess.run([self.loss, 
                                                self.accuracy,
                                                self.cm,
                                                self.merged_summary],
                                                feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                cm += cm_tmp

            l, acc = np.array([l, acc]) / n_batch

            summary = tf.Summary()
            summary.value.add(tag="Test/Accuracy", simple_value=acc)
            summary.value.add(tag="Test/Loss", simple_value=l)
            
            for i in range(self.NUM_LABELS):
                name = "Label_{}".format(lb_name[i])
                TP, TN, FP, FN = GetCMInfo(cm, i, self.NUM_LABELS)
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
            self.summary_test_writer.add_summary(s, step)

            print('  Validation loss: %.5f' % l)
            print('       Accuracy: %5.f%% \n   ' % (acc * 100))
            print('  Confusion matrix: \n')
            print_cm(cm, lb_name)
            confusion = tf.Variable(cm, name='confusion' )
            confusion_image = tf.reshape( tf.cast( confusion, tf.float32),
                                      [1, self.NUM_LABELS, self.NUM_LABELS, 1])
            tf.summary.image('confusion', confusion_image)
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", global_step=self.global_step)

    def train(self, DGTest):
        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.regularize_model()

        self.Saver()
        
        

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        print "self.global step", int(self.global_step.eval())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        begin = int(self.global_step.eval())
        print "begin", begin
        for step in range(begin, steps + begin):  
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            #pdb.set_trace()
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n ') % l
                self.Validation(DGTest, step)
        coord.request_stop()
        coord.join(threads)
    def predict(self, tensor):
        feed_dict = {self.input_node: tensor,
                     self.is_training: False}
        pred = self.sess.run(self.predictions,
                            feed_dict=feed_dict)
        return pred


if __name__== "__main__":

    parser = OptionParser()

#    parser.add_option("--gpu", dest="gpu", default="0", type="string",
#                      help="Input file (raw data)")
    parser.add_option("--tf_record", dest="TFRecord", type="string",
                      help="Where to find the TFrecord file")

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

    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5, help="dropout value to apply to the FC layers.")

    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")

    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")
    parser.add_option('--restore', dest="restore", type=str,
                      help="Folder from where to restore parameters.")
    (options, args) = parser.parse_args()

#    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

    TFRecord = options.TFRecord
    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    DROPOUT = options.dropout
    MEAN_FILE = options.mean_file 
    N_THREADS = options.THREADS



    LEARNING_RATE = options.lr
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str
    disk_size = options.TFRecord.split('.')[0].split('_')[1]
    ap = "first" if options.restore is None else "second"
    SAVE_DIR = SAVE_DIR + "__" + disk_size + "_" + ap 
    
    HEIGHT = 224 
    WIDTH = 224
    
    
    BATCH_SIZE = options.bs
    LRSTEP = "10epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch

    PATH = options.path


    HEIGHT = 212
    WIDTH = 212
    SIZE = (HEIGHT, WIDTH)

    N_TRAIN_SAVE = 100
 
    CROP = 4


    transform_list, transform_list_test = ListTransform()

    DG_TRAIN = DataGenMulti(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file=None)

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE

    DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)

    model = UNet3(TFRecord,    LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=3,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT)

    model.train(DG_TEST)
