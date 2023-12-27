# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:10:42 2022

@author: simon
"""

from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
import os
from util import config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from random import choice
#For general comparison. We do not include the user/item features extracted from text/images

class [Name](SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super([Name], self).readConfiguration()
        args = config.OptionConf(self.config['[Name]'])
            
    def initModel(self):
        super([Name], self).initModel()

    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()

        self.final_user_embeddings = user_embeddings+tf.sparse_tensor_dense_matmul(self.A,self.item_embeddings)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l, self.U, self.V = self.sess.run([train, loss, self.final_user_embeddings, self.item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
            self.ranking_performance(epoch)
        self.U,self.V = self.bestU,self.bestV
        
    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.item_embeddings])
        
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            #print(u)
            #return self.sess.run(self.test,feed_dict={self.u_idx:u})
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items