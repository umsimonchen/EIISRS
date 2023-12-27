#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import choice
from util import config
import tensorflow as tf

class MultiVAE(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MultiVAE, self).__init__(conf,trainingSet,testSet,fold)

    def encoder(self,x):
        ego = tf.nn.tanh(tf.matmul(x, self.weights['encoder'])+self.biases['encoder'])
        mean, logvar = tf.split(ego, [self.n_hidden, self.n_hidden], axis=1)
        stdvar = tf.exp(0.5*logvar)
        layer = mean + tf.multiply(tf.random_normal(tf.shape(stdvar)), stdvar)
        KL = tf.reduce_mean(tf.reduce_sum(0.5*(-logvar + tf.exp(logvar) + mean**2 -1), axis=1))
        return layer, KL

    def decoder(self,x):
        layer = tf.nn.tanh(tf.matmul(x, self.weights['decoder'])+self.biases['decoder'])
        return layer

    def next_batch(self):
        X = np.zeros((self.batch_size,self.num_items))
        uids = []
        positive = np.zeros((self.batch_size, self.num_items))
        negative = np.zeros((self.batch_size, self.num_items))
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        for n in range(self.batch_size):
            user = choice(userList)
            uids.append(self.data.user[user])
            vec = self.data.row(user)
            ratedItems, values = self.data.userRated(user)
            for item in ratedItems:
                iid = self.data.item[item]
                positive[n][iid]=1
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while ng in self.data.trainSet_u:
                    ng = choice(itemList)
                n_id = self.data.item[ng]
                negative[n][n_id]=1
            X[n]=vec
        return X,uids,positive,negative

    def readConfiguration(self):
        super(MultiVAE, self).readConfiguration()
        args = config.OptionConf(self.config['MultiVAE'])
        self.n_hidden = int(args['-nh'])

    def initModel(self):
        super(MultiVAE, self).initModel()
        self.negative_sp = 1
        initializer = tf.contrib.layers.xavier_initializer()
        self.X = tf.placeholder(tf.float32, [None, self.num_items])
        self.positive = tf.placeholder(tf.float32, [None, self.num_items])
        self.negative = tf.placeholder(tf.float32, [None, self.num_items])
        self.weights = {
            'encoder': tf.Variable(initializer([self.num_items, self.n_hidden*2])),
            'decoder': tf.Variable(initializer([self.n_hidden, self.num_items])),
        }
        self.biases = {
            'encoder': tf.Variable(initializer([self.n_hidden*2])),
            'decoder': tf.Variable(initializer([self.num_items])),
        }

    def trainModel(self):
        self.encoder_op, KL = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        y_pred = tf.multiply(self.decoder_op,self.X)
        y_pred = tf.maximum(1e-6,y_pred)
        y_positive = tf.multiply(self.positive,self.X)
        y_negative = tf.multiply(self.negative,self.X)
        self.loss = -tf.multiply(y_positive,tf.log(y_pred))-tf.multiply((y_negative),tf.log(1-y_pred))
        self.reg_loss = self.regU*(tf.nn.l2_loss(self.weights['encoder'])+tf.nn.l2_loss(self.weights['decoder'])+
                                   tf.nn.l2_loss(self.biases['encoder'])+tf.nn.l2_loss(self.biases['decoder']))
        self.reg_loss = self.reg_loss
        self.loss = tf.reduce_mean(self.loss) + self.reg_loss + KL
        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)


        for epoch in range(self.maxEpoch):
            batch_xs,users,positive,negative = self.next_batch()
            _, loss= self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_xs, 
                                                                      self.u_idx:users,self.positive:positive,self.negative:negative})
            print(self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"loss=", "{:.9f}".format(loss))
            #print y
            self.ranking_performance(epoch)
        print("Optimization Finished!")



    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            vec = self.data.row(u).reshape((1,self.num_items))
            uid = [self.data.user[u]]
            return self.sess.run(self.decoder_op,feed_dict={self.X:vec,self.u_idx:uid})[0]
        else:
            return [self.data.globalMean] * self.num_items


