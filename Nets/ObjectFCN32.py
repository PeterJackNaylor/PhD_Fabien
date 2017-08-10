from UNetBatchNorm_v2 import UNetBatchNorm




class FCN32(UNetBatchNorm):

    def init_vars(self):
        self.is_training = tf.placeholder(tf.bool)

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = self.N_FEATURES

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


        self.tfc7_conv5_weights = self.weight_xavier(2, 8 * n_features, 16 * n_features, "tconv5_4/")
        self.tfc7_conv5_biases = self.biases_const_f(0.1, 8 * n_features, "tconv5_4/")



        self.logits_weights = self.weight_xavier(1, 64 * n_features, 2, "logits/")
        self.logits_biases = self.biases_const_f(0.1, 2, "logits/")

        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")

        print('Model variables initialised')


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


        self.pool3_4 = self.max_pool(self.relu3_3, name="pool3_4")


        self.conv4_1 = self.conv_layer_f(self.pool3_4, self.conv4_1weights, "conv4_1/")
        self.relu4_1 = self.relu_layer_f(self.conv4_1, self.conv4_1biases, "conv4_1/")

        self.conv4_2 = self.conv_layer_f(self.relu4_1, self.conv4_2weights, "conv4_2/")
        self.relu4_2 = self.relu_layer_f(self.conv4_2, self.conv4_2biases, "conv4_2/")

        self.conv4_3 = self.conv_layer_f(self.relu4_2, self.conv4_3weights, "conv4_3/")
        self.relu4_3 = self.relu_layer_f(self.conv4_3, self.conv4_3biases, "conv4_3/")


        self.pool4_5 = self.max_pool(self.relu4_3, name="pool4_5")


        self.conv5_1 = self.conv_layer_f(self.pool4_5, self.conv5_1weights, "conv5_1/")

        self.relu5_1 = self.relu_layer_f(self.conv5_1, self.conv5_1biases, "conv5_1/")

        self.conv5_2 = self.conv_layer_f(self.relu5_1, self.conv5_2weights, "conv5_2/")
        self.relu5_2 = self.relu_layer_f(self.conv5_2, self.conv5_2biases, "conv5_2/")

        self.conv5_3 = self.conv_layer_f(self.relu5_2, self.conv5_3weights, "conv5_3/")
        self.relu5_3 = self.relu_layer_f(self.conv5_3, self.conv5_3biases, "conv5_3/")


        self.pool5_fc = self.max_pool(self.relu5_3, name="pool5_fc")


        self.fc6 = self.conv_layer_f(self.pool5_fc, self.fc6_weights, "fc6/", padding="VALID")
        self.fc6_relu = self.relu_layer_f(self.fc6, self.fc6_biases, "fc6/")
        self.fc6_dropout = self.DropOutLayer(self.fc6_relu, "fc6/")
        
        self.fc7 = self.conv_layer_f(self.fc6_relu, self.fc7_weights, "fc7/")
        self.fc7_relu = self.relu_layer_f(self.fc7, self.fc7_biases, "fc7/")
        self.fc7_dropout = self.DropOutLayer(self.fc7_relu, "fc7/")




        self.score_fr = self.conv_layer_f(self.fc7_dropout, self.score_fr_weights, "score_fr/", padding="VALID")
        score_fr = Conv(n.drop7, nout=num_output, ks=1, pad=0)


        self.conv_logit = self.conv_layer_f(self.fc7_dropout, self.logits_weights, "logits/")
        self.relu_logit = self.relu_layer_f(self.conv_logit, self.logits_biases, "logits/")
        self.last = self.relu_logit

        print('Model architecture initialised')