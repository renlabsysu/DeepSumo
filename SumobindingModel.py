# -*- coding: utf-8 -*-
# @Time    : 2019/7/9
# @Author  : CHEN Li and ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : SumobindingModel.py
# @Software: PyCharm
import tensorflow as tf
import math
import numpy as np
from sklearn import metrics
import copy


class CNN_SUMO:
    def __init__(self, all_id=None ,input_data=None,input_target=None,valid_data=None,valid_target=None,keep_prob=0.7,
                 bp_num=1, model_mode='train',mode='onehot',batch_size=20,epochs=100,learning_rate=0.01,
                 model_path=None,model_name="model",display_step=10,early_stop_thresh=0.000001,verbose=True,train_length=5):
        self.bp_num = bp_num  # unknown parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta = 0.0001  # didn't use
        self.batch_size = batch_size
        self.all_id=all_id  # of same length of input_data, mapping exactly to the input data's protein
        self.est = early_stop_thresh

        self.keep_prob_rate = keep_prob
        self.train_data= input_data
        self.train_target = input_target  # this will be leave as None if the model mode is "predict"
        self.valid_data = valid_data  # surveilance on the performance of model on each display step
        self.valid_target = valid_target

        self.dim = (train_length * 2 + 5) * 21
        self.dim2 = 30

        # well there will be a situation like this: we don't input train data
        if self.train_data is None:
            self.train_data = np.zeros([1, self.dim+self.dim2])
            # self.train_data = np.zeros([1, self.dim3])
            self.train_target = np.zeros([1,2])
        self.train_data_target = np.hstack([self.train_data,self.train_target])


        self.Size = self.train_data.shape  # the shape of input data
        self.sample_num, self.dim_all = self.Size[0], int(self.Size[1] / self.bp_num)
        # if mode == 'onehot':  # though we don't know what will be else, we leave it here in case there is some correction in the future
        self.total_dim = int(self.Size[1] / self.bp_num)

        self.train_data_target_cp = copy.deepcopy(self.train_data_target)
        self.pos_train = self.train_data_target_cp[np.where(self.train_data_target_cp[:, -2] == 1), :].reshape(-1,
                                                                                                               self.dim_all + 2)
        self.neg_train = self.train_data_target_cp[np.where(self.train_data_target_cp[:, -2] == 1), :].reshape(-1,
                                                                                                               self.dim_all + 2)



        self.onehot = self.train_data[:,0:self.dim]
        self.knn = self.train_data[:,self.dim:self.dim + self.dim2]

        self.epoch_train_err, self.epoch_valid_err = [], []

        self.model_mode = model_mode  # to train or to predict new data from saved model
        self.test_outs = None
        self.model_path = model_path
        self.sess = None
        self.display_step = display_step  # show result at each display_step when training model

        # all placeholders we need
        self.xs = tf.placeholder(tf.float32, [None, self.bp_num * self.dim], name="xs")  # onehot
        self.ys = tf.placeholder(tf.float32, [None, 2], name="ys")
        self.xs2 = tf.placeholder(tf.float32, [None, self.bp_num * self.dim2], name="xs2")  # knn
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # whether to show details of training process
        self.verbose = verbose
        self.model_name = model_name

    def model_initialize(self):  # TODO: this function is redundant! we will delete it later
        self.model = self.conf_model()
        self.config_trainer()
        # return  # actually we don't need to return anything

    def pick_batch(self,iteration,batch_data,rs_shape=False,new_shape=None):
        # inorder to calculate loss or for other purposes, rs_shape means whether we need to reshape the output
        # if true, then we need to set new_shape to a list of shape
        begin = iteration * self.batch_size
        end = (iteration+1) * self.batch_size
        if end >= self.sample_num:
            end = self.sample_num - 1
        if rs_shape is False:
            return batch_data[begin:end]
        else:
            return batch_data[begin:end].reshape(new_shape)

    def pick_batch_reg(self,iteration,batch_data,extra_pos,extra_neg,rs_shape=False,new_shape=None,balace_sp=False,close_ratio=1):
        # inorder to calculate loss or for other purposes, rs_shape means whether we need to reshape the output
        # if true, then we need to set new_shape to a list of shape
        # batch_data means something different from that of the above function. It contains both X and Y
        # close ratio is the ratio pos:neg that we want to pick
        begin = iteration * self.batch_size
        end = (iteration+1) * self.batch_size
        if end >= self.sample_num:
            end = self.sample_num - 1
        if rs_shape is False:
            x = batch_data[begin:end,0:self.dim_all]
            y = batch_data[begin:end,self.dim_all:]
        else:
            x = batch_data[begin:end,0:self.dim_all].reshape(new_shape)
            y = batch_data[begin:end,self.dim_all:]
        if balace_sp is False:
            pos_sp = list(y[:,0]).count(1)
            neg_sp = len(y[:,0]) - pos_sp
            # dif_pn = abs(pos_sp - neg_sp)
            if pos_sp < neg_sp:
                extra_pick = extra_pos
                dif_pn = int(neg_sp * close_ratio - pos_sp)  # to construct a sample with specific combination of pos and neg samples
            else:
                extra_pick = extra_neg
                dif_pn = int(pos_sp * close_ratio - neg_sp)
            if dif_pn > 0:
                np.random.shuffle(extra_pick)
                more_x,more_y = extra_pick[0:dif_pn,0:self.dim_all],extra_pick[0:dif_pn,self.dim_all:]
                x = np.vstack([x,more_x])
                y = np.vstack([y,more_y])
            return x, y
        else:
            return x, y

    def conf_model(self):
        x_matrix = tf.reshape(self.xs,[-1,int(self.dim/21),21])  # dim/21 * 21(row * col)
        x_matrix = tf.transpose(x_matrix,perm=[0,2,1])
        x_matrix = tf.reshape(x_matrix, [-1, int(self.bp_num * 21), int(self.dim / 21), 1])  # 21 * dim/21 (row * col)

        x_matrix2 = tf.reshape(self.xs2, [-1, self.bp_num, self.dim2, 1])  # original knn sequence, like a bargraph in the meaning of a graph

        # TODO: change strides or filters length (in col axis) to fit a better model
        # onehot
        hidden_conv1 = tf.layers.conv2d(inputs=x_matrix,filters=8,kernel_size=[21,5],strides=[self.bp_num*21,1],padding="same",
                                 activation=tf.nn.relu)  # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001)
        pool_conv2 = self.max_pool(inputs=hidden_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])

        hidden_conv2 = tf.layers.conv2d(inputs=pool_conv2, filters=16, kernel_size=[1, 3], strides=[1, 1],
                                        padding="same", activation=tf.nn.relu)
        pool_conv2 = self.max_pool(inputs=hidden_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])

        # knn
        hidden_conv21 = tf.layers.conv2d(inputs=x_matrix2, filters=8, kernel_size=[1, 3], strides=[1, 1],
                                        padding="same", activation=tf.nn.relu,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        pool_conv21 = self.max_pool(inputs=hidden_conv21, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])

        conv_all = tf.reshape(pool_conv2, [-1, pool_conv2.shape[1]*pool_conv2.shape[2]*pool_conv2.shape[3]])
        conv_all2 = tf.reshape(pool_conv21, [-1, pool_conv21.shape[1]*pool_conv21.shape[2]*pool_conv21.shape[3]])
        ################ zyr added in 190604, just try to get a better model ###########
        h_fc_drop = tf.concat([conv_all,conv_all2],1)
        ##############################################################
        # fully connected network
        first_unit = int((self.dim/21+self.dim2))
        second_unit = int(first_unit/4)

        fc4 = tf.layers.dense(inputs=h_fc_drop, units=first_unit, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))  # fully-connection can be more precise
        h_fc1_drop2 = tf.nn.dropout(fc4, self.keep_prob)

        fc4 = tf.layers.dense(inputs=h_fc1_drop2, units=second_unit, activation=tf.nn.relu)

        model = tf.layers.dense(inputs=fc4, units=2, activation=tf.nn.softmax)

        return model


    def run(self):  # train model and output a sess or make sess an attribute
        # model, loss and optimizer
        model = tf.reshape(self.conf_model(), [-1, 2], name="model_output")
        lossFunc = -tf.reduce_mean(self.ys * tf.log(model))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(lossFunc)

        # begin running or the so call training
        sess = tf.Session()
        # TODO:if it doesn't succeed, drop the following codes
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        total_batch = int(self.sample_num / self.batch_size)

        prev_c = 0
        best_val_auc = 0
        best_train_auc = 0
        for epochs in range(self.epochs):
            for i in range(total_batch):
                batch_xs = self.pick_batch(i,batch_data=self.train_data)
                batch_ys = self.pick_batch(i,batch_data=self.train_target)
                # batch_xs,batch_ys = self.pick_batch_reg(i,batch_data=self.train_data_target,extra_pos=self.pos_train,extra_neg=self.neg_train,close_ratio=1)
                train_data1 = batch_xs[:,0:self.dim]
                train_data2 = batch_xs[:,self.dim:(self.dim+self.dim2)]

                _, c = sess.run([optimizer, lossFunc], feed_dict={self.xs:train_data1,self.xs2:train_data2,
                                                                  self.ys: batch_ys,self.keep_prob: self.keep_prob_rate})
            if epochs % self.display_step == 0:
                # print("Epoch: %d" % (epochs + 1), "cost=", "{:.9f}".format(c))
                self.sess = sess
                if self.verbose is True:
                    if self.valid_data is not None and self.valid_target is not None:
                        vd1,vd2 = self.valid_data[:,0:self.dim], self.valid_data[:,self.dim:(self.dim2+self.dim)] # self.valid_data[:,(self.dim2+self.dim):(self.dim2+self.dim+self.dim3)]
                        # vd1,vd2 = self.valid_data[:,0:self.dim],self.valid_data[:,self.dim:(self.dim2+self.dim)]
                        pred = sess.run(model,feed_dict={self.xs:vd1,self.xs2:vd2,self.keep_prob:self.keep_prob_rate})
                        # pred = sess.run(model, feed_dict={self.xs:vd1,self.xs2:vd2, self.keep_prob: self.keep_prob_rate})
                        pred_train = sess.run(model,feed_dict={self.xs:self.onehot,self.xs2:self.knn,self.keep_prob:self.keep_prob_rate})
                        loss = -np.mean(self.valid_target * np.log(pred))
                        print("Epoch: %d" % (epochs + 1), "cost=", "{:.9f}".format(c),", Valid Data Loss:","{:.9f}".format(loss))
                        AUC_valid = self.GetAUCFromPredAndActual(self.valid_target,pred)
                        AUC_train = self.GetAUCFromPredAndActual(self.train_target,pred_train)
                        if AUC_valid > best_val_auc:
                            best_val_auc = AUC_valid
                            self.restore_model(path=self.model_path,modelname=self.model_name)  # save model in every display_step
                        print("AUC of validation data is:",AUC_valid,"and of training data is:",AUC_train)
                    else:
                        print("Epoch: %d" % (epochs + 1), "cost=", "{:.9f}".format(c))
                else:  # save the model with best auc in training data
                    pred_train = sess.run(model, feed_dict={self.xs: self.onehot,self.xs2:self.knn,
                                                            self.keep_prob: self.keep_prob_rate})
                    AUC_train = self.GetAUCFromPredAndActual(self.train_target, pred_train)
                    if best_val_auc < AUC_train:
                        self.restore_model(path=self.model_path,modelname=self.model_name)  # save model in every display_step

            if abs(prev_c-c) <= self.est:
                break
        print("Optimization Finished!")

        # inheritage the info above
        coord.request_stop()
        coord.join(threads=threads)

    def restore_model(self, path=None,modelname=None):
        # path should contains os.sep in the end of the string
        self.saver = tf.train.Saver(save_relative_paths=True)
        if path is None:
            path = self.model_path
        if modelname is None:
            modelname = self.model_name
        if self.model_mode != "train":
            # self.saver.restore(self.sess, PATH+"model.ckpt")  # TODO: high chance to change ckpt into meta
            self.saver.restore(self.sess, path + modelname + ".meta")  # actually this sentence if of no use or it is useless
        else:
            self.saver.save(self.sess, path + modelname)

    def predict(self,other_data=None,model_path=None,modelname=None):
        tf.reset_default_graph()  # in case newly input graph contaminates the last one
        if model_path is None:
            model_path = self.model_path
        if other_data is None:
            other_data = self.train_data
        # divide other data into three part
        onehot_data = other_data[:,0:self.dim]
        knn_data = other_data[:,self.dim:(self.dim+self.dim2)]

        if modelname is None:
            modelname = "model"

        # get saved sess and predict
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_saver = tf.train.import_meta_graph(model_path + modelname + ".meta")
            new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
            # new_saver.restore(sess, save_path=model_path)
            graph = tf.get_default_graph()
            xs = graph.get_tensor_by_name("xs:0")
            xs2 = graph.get_tensor_by_name("xs2:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            model_out = graph.get_tensor_by_name("model_output:0")

            pred = sess.run([model_out], feed_dict={xs:onehot_data,xs2:knn_data,keep_prob: self.keep_prob_rate})
        return pred[0]


    def GetAUCFromPredAndActual(self,true_label,pred_score):
        # presume that both label and pred score are 2 dimension
        try:
            pred_score = [i[0] for i in pred_score]
        except:
            pred_score = pred_score
        true_score = [i[0] for i in true_label]
        AUC = metrics.roc_auc_score(y_true=true_score, y_score=pred_score)
        return AUC

    def GetStatisticPred(self,pred,y_new,cutoff=0.5):
        ppn = []
        for i in pred:
            if isinstance(i,float) or isinstance(i,int):
                i = [i,0.5]
            if i[0] >= cutoff:
                ppn.append(1)
            else:
                ppn.append(0)
        tp, tn, fp, fn = 0, 0, 0, 0
        for idx, j in enumerate(ppn):
            if j == 1 and y_new[idx][0] == 1:
                tp += 1
            elif j == 1 and y_new[idx][0] == 0:
                fp += 1
            elif j == 0 and y_new[idx][0] == 0:
                tn += 1
            elif j == 0 and y_new[idx][0] == 1:
                fn += 1
        ac = (tp + tn) / (tp + tn + fp + fn)
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
        print("ac,sn,sp,mcc,tp,fp,tn,fn:",ac, sn, sp, mcc,tp,fp,tn,fn)

    def filter_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, inputs, filter, strides):
        return tf.nn.conv2d(inputs, filter=filter, strides=strides, padding='SAME')

    def max_pool(self, inputs, ksize, strides):
        return tf.nn.max_pool(value=inputs, ksize=ksize, strides=strides, padding='SAME')

    def relu(self, inputs):
        return tf.nn.relu(inputs)