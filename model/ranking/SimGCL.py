# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:51:49 2022

@author: simon
"""

from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util import config
from util.loss import bpr_loss
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Note: the pytorch implementation of SimGCL can be found at https://github.com/Coder-Yu/SELFRec


class SimGCL(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SimGCL, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(SimGCL, self).readConfiguration()
        args = config.OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])

    def LightGCN_encoder(self,emb,adj,n_layers):
        all_embs = [] #no the first random initialized embedding
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.num_users, self.num_items], 0)

    def perturbed_LightGCN_encoder(self,emb,adj,n_layers):
        all_embs = [] #no the first random initialized embedding
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb) #norm+adj is sparse, but emb is dense
            random_noise = tf.random.uniform(emb.shape)
            emb += tf.multiply(tf.sign(emb),tf.nn.l2_normalize(random_noise, 1)) * self.eps #eps control the magnitude of noise
            all_embs.append(emb)
        all_embs = tf.reduce_mean(all_embs, axis=0)
        return tf.split(all_embs, [self.num_users, self.num_items], 0)

    def initModel(self):
        super(SimGCL, self).initModel()
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.emb_size]))
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.emb_size]))
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        #adjaceny matrix
        self.norm_adj = self.create_joint_sparse_adj_tensor()
        #encoding
        self.main_user_embeddings, self.main_item_embeddings = self.LightGCN_encoder(ego_embeddings,self.norm_adj,self.n_layers) #normal GCN
        self.perturbed_user_embeddings1, self.perturbed_item_embeddings1 = self.perturbed_LightGCN_encoder(ego_embeddings,self.norm_adj, self.n_layers)
        self.perturbed_user_embeddings2, self.perturbed_item_embeddings2 = self.perturbed_LightGCN_encoder(ego_embeddings, self.norm_adj, self.n_layers)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def calc_cl_loss(self):
        p_user_emb1 = tf.nn.embedding_lookup(self.perturbed_user_embeddings1, tf.unique(self.u_idx)[0])
        p_item_emb1 = tf.nn.embedding_lookup(self.perturbed_item_embeddings1, tf.unique(self.v_idx)[0])
        p_user_emb2 = tf.nn.embedding_lookup(self.perturbed_user_embeddings2, tf.unique(self.u_idx)[0])
        p_item_emb2 = tf.nn.embedding_lookup(self.perturbed_item_embeddings2, tf.unique(self.v_idx)[0])
        # group contrast
        normalize_emb_user1 = tf.nn.l2_normalize(p_user_emb1, 1)
        normalize_emb_user2 = tf.nn.l2_normalize(p_user_emb2, 1)
        normalize_emb_item1 = tf.nn.l2_normalize(p_item_emb1, 1)
        normalize_emb_item2 = tf.nn.l2_normalize(p_item_emb2, 1)
        #pos means alignment with inner product, ttl means uniformity with inner product as well
        pos_score_u = tf.reduce_sum(tf.multiply(normalize_emb_user1, normalize_emb_user2), axis=1) #only calculate the score of a node itself, serve as positive pair
        pos_score_i = tf.reduce_sum(tf.multiply(normalize_emb_item1, normalize_emb_item2), axis=1)
        ttl_score_u = tf.matmul(normalize_emb_user1, normalize_emb_user2, transpose_a=False, transpose_b=True) #take all other node into account
        ttl_score_i = tf.matmul(normalize_emb_item1, normalize_emb_item2, transpose_a=False, transpose_b=True)
        
        pos_score_u = tf.exp(pos_score_u / 0.2) #temperature=0.2
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u / 0.2), axis=1)
        pos_score_i = tf.exp(pos_score_i / 0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i / 0.2), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score_u / ttl_score_u)) - tf.reduce_sum(tf.log(pos_score_i / ttl_score_i))

        #hybrid contrast
        # emb_merge1 = tf.concat([p_user_emb1, p_item_emb1], axis=0)
        # emb_merge2 = tf.concat([p_user_emb2, p_item_emb2], axis=0)
        # normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        # normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
        # pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        # ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)
        # pos_score = tf.exp(pos_score /0.2)
        # ttl_score = tf.reduce_sum(tf.exp(ttl_score /0.2), axis=1)
        # cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        return self.cl_rate*cl_loss

    def trainModel(self):
        #main task: recommendation
        rec_loss = bpr_loss(self.batch_user_emb,self.batch_pos_item_emb,self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(self.batch_neg_item_emb))
        #CL task
        self.cl_loss = self.calc_cl_loss()
        loss = rec_loss+self.cl_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l, rec_l, cl_l = self.sess.run([train, loss, rec_loss, self.cl_loss],
                                                   feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'total_loss:',l, 'rec_loss:', rec_l,'cl_loss',cl_l)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.ranking_performance(epoch)
            
            # early stop
            if self.earlyStop == 50:
                break
        self.U,self.V = self.bestU,self.bestV

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items