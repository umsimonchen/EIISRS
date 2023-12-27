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

class DiffNetLG(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(DiffNetLG, self).readConfiguration()
        args = config.OptionConf(self.config['DiffNetLG'])
        self.n_layers = int(args['-n_layer'])
    
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]] # row index
            col += [self.data.user[pair[1]]] # column index
            entries += [1.0] # corresponding value of row and column in the same position 
        original = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32) #m*m
        alpha = original.multiply(1.0/original.sum(axis=1).reshape(-1,1))
        return self.adj_to_sparse_tensor(original), self.adj_to_sparse_tensor(alpha)

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        original = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32) #m*n
        ratingMatrix = original.multiply(1.0/original.sum(axis=1).reshape(-1,1)) #m*n
        ratedMatrix = original.T.multiply(1.0/original.T.sum(axis=1).reshape(-1,1)) #n*m
        return self.adj_to_sparse_tensor(ratingMatrix), self.adj_to_sparse_tensor(ratedMatrix), self.adj_to_sparse_tensor(original.T) #n*m
    
    def initModel(self):
        super(DiffNetLG, self).initModel()
        
    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['gating'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W') #d*d
        self.weights['gating_bias'] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b') #d*1
        
        self.user_embeddings = tf.multiply(self.user_embeddings,tf.nn.sigmoid(tf.matmul(self.user_embeddings,self.weights['gating'])+self.weights['gating_bias'])) 
        
        all_user_embeddings = [self.user_embeddings]
        all_item_embeddings = [self.item_embeddings]
        beta, eta, original_item = self.buildSparseRatingMatrix() #η,β
        original_social, alpha = self.buildSparseRelationMatrix() #α
        original_social = tf.sparse.to_dense(original_social)
        alpha = tf.sparse.to_dense(alpha)
        original_item = tf.sparse.to_dense(original_item, validate_indices=False)
        popularity_item = tf.reduce_sum((original_item + 1) / (tf.reduce_sum(original_item) + self.num_items), axis=1, keepdims=True) #n*1
        
        for k in range(self.n_layers):
            new_item_embeddings = tf.sparse_tensor_dense_matmul(eta, all_user_embeddings[k])
            embedding_p = tf.matmul(alpha, all_user_embeddings[k])
            embedding_p = tf.math.l2_normalize(embedding_p, axis=1)
            explicit_embedding_q = tf.sparse_tensor_dense_matmul(beta, all_item_embeddings[k])
            explicit_embedding_q = tf.math.l2_normalize(explicit_embedding_q)
            
            user_similarity = tf.matmul(all_user_embeddings[k], all_user_embeddings[k], transpose_b=True) #m*m
            user_similarity = 1 / (1 + tf.math.exp(-user_similarity)) #m*m
            alpha = tf.cast((user_similarity + original_social > 0.9), tf.float32)
            alpha = alpha / tf.reduce_sum(alpha, axis=1)
            
            tao = tf.matmul(tf.math.l2_normalize(all_item_embeddings[k], axis=1), tf.math.l2_normalize(all_user_embeddings[k], axis=1), transpose_b=True) #n*m
            tao = tf.multiply(tao, tf.tile(popularity_item, [1, self.num_users])) #n*m
            implicit_embedding_q = tf.matmul(tao, all_item_embeddings[k], transpose_a=True)
            implicit_embedding_q = tf.math.l2_normalize(implicit_embedding_q)
            
            embedding_q = (explicit_embedding_q + implicit_embedding_q) / 2
            
            all_user_embeddings += [(all_user_embeddings[k] + embedding_p + embedding_q) / 3]
            all_item_embeddings += [new_item_embeddings]
        
        self.final_user_embeddings = self.user_embeddings
        self.final_item_embeddings = self.item_embeddings
        for k in range(1, self.n_layers):
            self.final_user_embeddings = tf.concat([self.final_user_embeddings, all_user_embeddings[k]], axis=1)
            self.final_item_embeddings = tf.concat([self.final_item_embeddings, all_item_embeddings[k]], axis=1)
            
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        writer = tf.summary.FileWriter('C:/Users/simon/Desktop/QRec-master/tensorboard', self.sess.graph)
        loss_summ = tf.summary.scalar('loss', loss)
        
        self.prec = tf.placeholder(dtype=tf.float32)
        self.reca = tf.placeholder(dtype=tf.float32)
        self.f1 = tf.placeholder(dtype=tf.float32)
        self.ndcg = tf.placeholder(dtype=tf.float32)
        
        a = tf.summary.scalar('precision', self.prec)
        b = tf.summary.scalar('recall', self.reca)
        c = tf.summary.scalar('f1', self.f1)
        d = tf.summary.scalar('ndcg', self.ndcg)
        merge_summ = tf.summary.merge([a,b,c,d])
        
        batch_round = (self.train_size // self.batch_size) + 1
        for epoch in range(self.maxEpoch):
            self.tensorboard = True
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l, self.U, self.V, s = self.sess.run([train, loss, self.final_user_embeddings, self.final_item_embeddings, loss_summ],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
                
                writer.add_summary(s, epoch*batch_round + n)
                measure = self.ranking_performance(epoch)
                
                writer.add_summary(self.sess.run(merge_summ,feed_dict={self.prec: float(measure[1].strip().split(':')[1]),
                                                                self.reca: float(measure[2].strip().split(':')[1]),
                                                                self.f1: float(measure[3].strip().split(':')[1]), 
                                                                self.ndcg: float(measure[4].strip().split(':')[1])
                                                                }), epoch*batch_round + n)
                
            self.tensorboard = False
            self.ranking_performance(epoch)
        writer.close()
        self.U,self.V = self.bestU,self.bestV
        
    def saveModel(self):
        self.bestU, self.bestV = self.U, self.V
        
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            #print(u)
            #return self.sess.run(self.test,feed_dict={self.u_idx:u})
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items