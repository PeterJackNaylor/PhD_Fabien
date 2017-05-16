import tensorflow as tf


XAVIER = tf.contrib.layers.xavier_initializer_conv2d()



IMAGE_SIZE_IN =
IMAGE_SIZE_OUT = 
NUM_LABEL =  
DIM = 
CONSTANT_INIT = 0.1

def InputLayer(ImageSizeIn, ImageSizeOut, Dim, NumLabel, Weight=False, Summary = True):
    with tf.name_scope('input'):
        # None -> batch size can be any size
        x_ = tf.placeholder(tf.float32, shape=[None, ImageSizeIn, ImageSizeIn, Dim], name="image-input") 
        y_ = tf.placeholder(tf.float32, shape=[None, ImageSizeOut, ImageSizeOut, NumLabel], name="label-input")

        if Summary:
            tf.summary.image("Input" ,x_, 3)
            tf.summary.image("Label", y_)
        
        if Weight:
            w_ = tf.placeholder(tf.float32, shape=[None, ImageSizeOut, ImageSizeOut, NumLabel], name="weight-input")
            return x_, y_, w_
        else:
            return x_, y_


def ConvLayer(Input, ChannelsIn, ChannelsOut, ks, Name, padding, Summary = True):
    with tf.name_scope(Name):
        #w_ = tf.Variable(tf.zeros([ks,ks, ChannelsIn, ChannelsOut]), name="W")
        #b_ = tf.Variable(tf.zeros([ChannelsOut]), name="B")
        kernel_shape = [ks, ks, ChannelsIn, ChannelsOut]
        
        w_ = tf.get_variable("W", kernel_shape, 
                    initializer=XAVIER)
        
        b_ = tf.get_variable("B", [ChannelsOut],
                    initializer=tf.constant_initializer(CONSTANT_INIT))

        conv = tf.nn.conv2d(Input, w, strides=[1,1,1,1], padding=padding)
        act = tf.nn.relu(conv + b)

        if Summary:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

        return act

def ConvLayerWithoutRelu(Input, ChannelsIn, ChannelsOut, ks, Name, padding, Summary = True):
    with tf.name_scope(Name):
        kernel_shape = [ks, ks, ChannelsIn, ChannelsOut]
        
        w_ = tf.get_variable("W", kernel_shape, 
                    initializer=XAVIER)
        if Summary:
            tf.summary.histogram("weights", w)

        return tf.nn.conv2d(Input, w, strides=[1,1,1,1], padding=padding)



def TransposeConvLayer(Input, ChannelsIn, ChannelsOut, Outsize, ks, Name, padding, Summary = True):
    with tf.name_scope(Name):
        kernel_shape = [ks, ks, ChannelsIn, ChannelsOut]
        
        w_ = tf.get_variable("W", kernel_shape, 
                    initializer=XAVIER)
        
        b_ = tf.get_variable("B", [ChannelsOut],
                    initializer=tf.constant_initializer(CONSTANT_INIT))

        OutputShape = np.array([None, Outsize, Outsize, ChannelsOut])
        trans_conv = tf.nn.conv2d_transpose(Input, w, output_shape=OutputShape,
                                            strides=[1,1,1,1], padding=padding)
        act = tf.nn.relu(trans_conv + b)

        if Summary:
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

        return act


def CropAndMerge(Input1, Input2, Size1, Size2, Channels, Name):
    ### Size1 > Size2
    with tf.name_scope(Name):
        assert Size1 > Size2 
        diff = (Size1 - Size2) / 2

        crop = tf.slice(Input1, [0, diff, diff, 0], [-1, Size2, Size2, Channels])
        concat = tf.concat([Input2, crop], axis=3)
        return concat



tf.summary.scalar('cross_entropy', xent)