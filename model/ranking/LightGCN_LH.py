# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:23:10 2023

@author: simon
"""

from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util.loss import bpr_loss
from util.config import OptionConf
from scipy.sparse import coo_matrix
from scipy import sparse
import numpy as np

class LightGCN_LH(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN_LH, self).__init__(conf,trainingSet,testSet,fold)
        args = OptionConf(self.config['LightGCN_LH'])
        self.n_layers = int(args['-n_layer'])
    
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # asymmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        #generate high pass matrix
        multiHopMatrix = ratingMatrix
        for i in range(9):
            multiHopMatrix = multiHopMatrix.dot(ratingMatrix.T) #m*m
            multiHopMatrix = multiHopMatrix.dot(ratingMatrix) #m*n
        highMatrix = (multiHopMatrix<1).astype(dtype=np.float32)
        normHighMatrix = highMatrix
        normHighMatrix = normHighMatrix / np.sqrt(highMatrix.sum(axis=1).reshape(-1,1))
        normHighMatrix = normHighMatrix.T / np.sqrt(highMatrix.sum(axis=0).reshape(-1,1))
        return normHighMatrix.T
    
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def initModel(self):
        super(LightGCN_LH, self).initModel()
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        low_adj = self.create_joint_sparse_adj_tensor() #(m+n)*(m+n)
        high_adj = self.buildSparseRatingMatrix() #m*n
        high_adj = sparse.csr_matrix(high_adj)
        high_adj = self.adj_to_sparse_tensor(high_adj) 
        high_adj = tf.sparse.to_dense(high_adj)
        
        #low_pass conv
        all_embeddings_low = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(low_adj,ego_embeddings)
            all_embeddings_low += [ego_embeddings]
        all_embeddings_low = tf.reduce_mean(all_embeddings_low, axis=0)
        
        #high_pass cov
        all_user_embeddings_high = [self.user_embeddings]
        all_item_embeddings_high = [self.item_embeddings]
        for k in range(self.n_layers):
            new_user_embeddings = all_user_embeddings_high[k] - tf.matmul(high_adj, all_item_embeddings_high[k])
            new_item_embeddings = all_item_embeddings_high[k] - tf.matmul(high_adj, all_user_embeddings_high[k], transpose_a=True)
            all_user_embeddings_high += [new_user_embeddings]
            all_item_embeddings_high += [new_item_embeddings]
        
        self.final_user_embeddings_high = tf.reduce_mean(all_user_embeddings_high, axis=0)
        self.final_item_embeddings_high = tf.reduce_mean(all_item_embeddings_high, axis=0)
        self.final_user_embeddings_low, self.final_item_embeddings_low = tf.split(all_embeddings_low, [self.num_users, self.num_items], 0)
        
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        #low graph embedding gather
        self.batch_neg_item_emb_low = tf.nn.embedding_lookup(self.final_item_embeddings_low, self.neg_idx) #n1
        self.batch_user_emb_low = tf.nn.embedding_lookup(self.final_user_embeddings_low, self.u_idx) #u1
        self.batch_pos_item_emb_low = tf.nn.embedding_lookup(self.final_item_embeddings_low, self.v_idx) #p2
        #high graph embeddings gather
        self.batch_neg_item_emb_high = tf.nn.embedding_lookup(self.final_item_embeddings_high, self.neg_idx) #n2
        self.batch_user_emb_high = tf.nn.embedding_lookup(self.final_user_embeddings_high, self.u_idx) #u2
        self.batch_pos_item_emb_high = tf.nn.embedding_lookup(self.final_item_embeddings_high, self.v_idx) #p2
        
        self.test = tf.reduce_sum(tf.multiply(self.batch_user_emb_low+self.batch_user_emb_high, self.final_item_embeddings_low+self.final_item_embeddings_high), 1)

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_user_emb_low, self.batch_pos_item_emb_low, self.batch_neg_item_emb_low)
        rec_loss += bpr_loss(self.batch_user_emb_low, self.batch_pos_item_emb_low, self.batch_neg_item_emb_high)
        rec_loss += bpr_loss(self.batch_user_emb_low, self.batch_pos_item_emb_high, self.batch_neg_item_emb_low)
        rec_loss += bpr_loss(self.batch_user_emb_low, self.batch_pos_item_emb_high, self.batch_neg_item_emb_high)
        rec_loss += bpr_loss(self.batch_user_emb_high, self.batch_pos_item_emb_low, self.batch_neg_item_emb_low)
        rec_loss += bpr_loss(self.batch_user_emb_high, self.batch_pos_item_emb_low, self.batch_neg_item_emb_high)
        rec_loss += bpr_loss(self.batch_user_emb_high, self.batch_pos_item_emb_high, self.batch_neg_item_emb_low)
        rec_loss += bpr_loss(self.batch_user_emb_high, self.batch_pos_item_emb_high, self.batch_neg_item_emb_high)
        rec_loss /= 8
        
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb_low+self.batch_user_emb_high) + tf.nn.l2_loss(self.batch_pos_item_emb_low+self.batch_pos_item_emb_high) + tf.nn.l2_loss(
                self.batch_neg_item_emb_low+self.batch_neg_item_emb_high))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(rec_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, rec_loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
        self.U, self.V = self.sess.run([self.final_user_embeddings_low+self.final_user_embeddings_high, self.final_item_embeddings_low+self.final_item_embeddings_high])   

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items