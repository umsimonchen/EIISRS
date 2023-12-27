# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Wed Dec 14 16:10:42 2022
# 
# @author: simon
# """
# 
# from base.graphRecommender import GraphRecommender
# from base.socialRecommender import SocialRecommender
# import tensorflow as tf
# from scipy.sparse import coo_matrix
# import numpy as np
# import os
# from util import config
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from math import sqrt
# from util.loss import bpr_loss
# #For general comparison. We do not include the user/item features extracted from text/images
# 
# os.environ["PYTHONHASHSEED"] = str(0)
# tf.reset_default_graph()
# tf.compat.v1.set_random_seed(0)
# np.random.seed(0)
# 
# class EISR(SocialRecommender,GraphRecommender):
#     def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
#         GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
#         SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
# 
#     def readConfiguration(self):
#         super(EISR, self).readConfiguration()
#         args = config.OptionConf(self.config['EISR'])
#         self.n_layers = int(args['-n_layer'])
#         self.num_neighbors = int(args['-num_neighbor'])
#         self.tau = float(args['-tau'])
#             
#     def initModel(self):
#         super(EISR, self).initModel()
#     
#     def buildSparseRelationMatrix(self):
#         row, col, entries = [], [], []
#         for pair in self.social.relation:
#             row += [self.data.user[pair[0]]]
#             col += [self.data.user[pair[1]]]
#             entries += [1.0] # here we normalize
#         AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
#         return AdjacencyMatrix
#     
#     def adj_to_sparse_tensor(self,adj):
#         adj = adj.tocoo()
#         indices = np.mat(list(zip(adj.row, adj.col)))
#         adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
#         return adj
#     
#     def buildExplicitImplicitMatrix(self):
#         social_graph = self.buildSparseRelationMatrix() #m*m
#         hetero_graph = self.create_joint_sparse_adj_tensor() #(m+n)*(m+n)
#         social_graph = self.adj_to_sparse_tensor(social_graph)
#         return social_graph, hetero_graph
#     
#     def gumbel_softmax(self, logits, temperature=0.2):
#         eps = 1e-10
#         u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1) #k*m
#         gumbel_noise = -tf.log(-tf.log(u + eps) + eps) #k*m
#         y = tf.log(logits + eps) + gumbel_noise #k*m
#         return tf.nn.softmax(y / temperature) #k*m
#     
#     def trainModel(self):
#         self.weights = {}
#         initializer = tf.contrib.layers.xavier_initializer()
#         original_social_graph, hetero_graph = self.buildExplicitImplicitMatrix()
#         original_social_graph = tf.sparse.to_dense(tf.sparse.reorder(original_social_graph)) #prepared for sampleed new graph
#         self.implicit_social_graph = tf.placeholder(tf.float32, shape=[self.num_users, self.num_users])
#         
#         #joint sample social graph
#         social_graph = original_social_graph + self.implicit_social_graph
#         # social_graph = tf.multiply(original_social_graph + self.implicit_social_graph, tf.random.uniform(shape=[self.num_users, self.num_users]))
#         # social_graph = tf.cast(social_graph>=(self.num_neighbors/(tf.reduce_sum(original_social_graph, axis=1) + self.num_neighbors)), tf.float32)
#         social_graph = tf.math.divide_no_nan(social_graph, tf.reduce_sum(social_graph, axis=1))
#         
#         #Training parameter
#         self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U') # m*d
#         self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') # n*d
#         for i in ['Social', 'Hetero']:
#             self.weights['gating%s'%i] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='gatedWeight%s'%i)
#             self.weights['gating_bias%s'%i] = tf.Variable(initializer([1, self.emb_size]), name='gatedWeightBias%s'%i)
#         self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
#         self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
#         self.weights['social2item'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='social2item')
#         for k in range(self.n_layers):
#             self.weights['vae_encoder%d'%k] = tf.Variable(initializer([self.emb_size, 2*self.emb_size]), name='encoder%d'%k)
#             self.weights['vae_decoder%d'%k] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='decoder%d'%k)
#     
#         def channel_attention(channel_embeddings):
#             weights = [] #k*m
#             for embedding in channel_embeddings:
#                 weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), axis=1)) #m
#             score = tf.nn.softmax(tf.transpose(weights), axis=1) #m*k
#             mixed_embeddings = 0 #broadcasting
#             for i in range(len(weights)):
#                 mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i]))) #m*d
#             return mixed_embeddings,score
#         
#         #Self-gating
#         gating_social = tf.nn.sigmoid(tf.matmul(self.user_embeddings, self.weights['gatingSocial']) + self.weights['gating_biasSocial'])
#         user_embeddings_social = tf.multiply(self.user_embeddings, gating_social)
#         gating_hetero = tf.nn.sigmoid(tf.matmul(self.user_embeddings, self.weights['gatingHetero']) + self.weights['gating_biasHetero'])
#         user_embeddings_hetero = tf.multiply(self.user_embeddings, gating_hetero)
#         
#         all_user_embeddings_social = [user_embeddings_social]
#         all_sample_user_embeddings_social = []
#         all_user_embeddings_hetero = [user_embeddings_hetero]
#         all_item_embeddings = [self.item_embeddings]
#         rec_loss = 0
#         kl_loss = 0
#         
#         #GCN
#         for k in range(self.n_layers):
#             #Social
#             user_embeddings_social = tf.matmul(social_graph, all_user_embeddings_social[k])
#             
#             #VAE for social-encoder
#             #mean_logvar = tf.nn.tanh(tf.matmul(user_embeddings_social, self.weights['vae_encoder%d'%k]))
#             mean_logvar = tf.matmul(user_embeddings_social, self.weights['vae_encoder%d'%k])
#             mean, logvar = tf.split(mean_logvar, [self.emb_size, self.emb_size], axis=1)
#             sample_user_embeddings_social = mean + tf.exp(logvar/2) * tf.random.normal(tf.shape(logvar))
#             all_user_embeddings_social += [tf.math.l2_normalize(user_embeddings_social, axis=1)]
#             all_sample_user_embeddings_social += [tf.math.l2_normalize(sample_user_embeddings_social, axis=1)]
#             #VAE for social-decoder
#             reconstructed_user_embeddings_social = tf.matmul(sample_user_embeddings_social, self.weights['vae_decoder%d'%k])
#             rec_loss += tf.reduce_sum(tf.reduce_mean(tf.math.square(user_embeddings_social - reconstructed_user_embeddings_social), axis=1))
#             kl_loss += tf.reduce_sum(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1), axis=0)
#             
#             #Hetero
#             mixed_embeddings_hetero = tf.sparse_tensor_dense_matmul(hetero_graph, tf.concat([all_user_embeddings_hetero[k], all_item_embeddings[k]], axis=0))
#             user_embeddings_hetero, item_embeddings = tf.split(mixed_embeddings_hetero, [self.num_users, self.num_items], axis=0)
#             all_user_embeddings_hetero += [tf.math.l2_normalize(user_embeddings_hetero, axis=1)]
#             all_item_embeddings += [tf.math.l2_normalize(item_embeddings, axis=1)]
#             
#         user_embeddings_hetero = tf.reduce_sum(all_user_embeddings_hetero, axis=0)
#         item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0)
#         
#         self.final_item_embeddings = item_embeddings
#         
#         #graph layer-wise attention
#         sample_user_embeddings_social, score = channel_attention(all_sample_user_embeddings_social)
#         #self.final_user_embeddings = tf.math.l2_normalize(tf.nn.tanh(tf.matmul(sample_user_embeddings_social, self.weights['social2item'])), axis=1) + user_embeddings_hetero
#         self.final_user_embeddings = sample_user_embeddings_social + user_embeddings_hetero
#         
#         #gumbel sample new relations
#         self.u_segment = tf.placeholder(tf.int32)
#         self.weights['selector'] = tf.Variable(initializer([self.num_neighbors, self.num_users]), name='selector')
#         user_features = tf.matmul(user_embeddings_hetero[self.u_segment:self.u_segment+100], user_embeddings_hetero, transpose_b=True)
#         def getNewNeighbor(embedding):
#             alpha = tf.nn.softmax(tf.multiply(embedding, self.weights['selector']))
#             alpha = self.gumbel_softmax(alpha, self.tau)
#             multi_hot_vector = tf.reduce_sum(alpha, axis=0)
#             return multi_hot_vector
#         implicit_social_graph = tf.vectorized_map(fn=lambda em:getNewNeighbor(em), elems=user_features)
#         new_old_mask = tf.math.logical_not(original_social_graph[self.u_segment:self.u_segment+100]>0) #m+*m 
#         implicit_social_graph = tf.where(new_old_mask, implicit_social_graph*1.0, implicit_social_graph*0.0) #totally new relations 
#         implicit_social_graph = tf.cast(implicit_social_graph>=tf.expand_dims(tf.math.top_k(implicit_social_graph, self.num_neighbors)[0][:,-1], axis=-1), tf.float32) #m+*m, top-k new neighbors
#         padding = tf.zeros(shape=(self.num_users, self.num_users))
#         implicit_social_graph = tf.concat([padding[:self.u_segment], implicit_social_graph, padding[self.u_segment+100:]], axis=0)
#         
#         #BPR
#         self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
#         self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
#         self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
#         self.v_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
#         self.test = tf.reduce_sum(tf.multiply(self.u_embedding, item_embeddings), 1)
#         
#         main_loss = bpr_loss(self.u_embedding, self.v_embedding, self.neg_item_embedding)
#         
#         loss = main_loss + self.regU * (
#                     tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
#                     tf.nn.l2_loss(self.neg_item_embedding)) + 0.01 * kl_loss + 0.1 * rec_loss
#         
#         opt = tf.train.AdamOptimizer(self.lRate)
#         train = opt.minimize(loss)
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#         self.no_improve = {i: 0 for i in range(50)}
#         self.imp_graph = np.zeros((self.num_users, self.num_users), dtype=np.float32)
#         for epoch in range(self.maxEpoch):
#             for n, batch in enumerate(self.next_batch_pairwise()):
#                 user_idx, i_idx, j_idx = batch
#                 u_i = np.random.randint(0, self.num_users)
#                 _, l, k_l, r_l, self.imp_graph, att_score = self.sess.run([train, loss, kl_loss, rec_loss, implicit_social_graph, score],
#                                      feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.u_segment: u_i, self.implicit_social_graph: self.imp_graph})
#                 print('training:', epoch + 1, 'batch', n, 'loss:', l, 'k_loss:', k_l, 'rec_loss:', r_l)
#             self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings],feed_dict={self.implicit_social_graph: self.imp_graph})
#             self.ranking_performance(epoch)
#             
#             #early stop
#             if self.earlyStop == 50:
#                 print(self.no_improve)
#                 break
#             
#         self.U,self.V = self.bestU,self.bestV
#         
#     def saveModel(self):
#         #feed_dict used for loss funtion training, w/o it still can generate final_user/item embeddings
#         self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings], feed_dict={self.implicit_social_graph: self.imp_graph})
#         
#     def predictForRanking(self, u):
#         'invoked to rank all the items for the user'
#         if self.data.containsUser(u):
#             u = self.data.getUserId(u)
#             #print(u)
#             #return self.sess.run(self.test,feed_dict={self.u_idx:u})
#             return self.V.dot(self.U[u])
#         else:
#             return [self.data.globalMean] * self.num_items
# =============================================================================
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
import os
from util import config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from math import sqrt
#For general comparison. We do not include the user/item features extracted from text/images

os.environ["PYTHONHASHSEED"] = str(0)
tf.reset_default_graph()
tf.compat.v1.set_random_seed(0)
np.random.seed(0)

class EISR(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(EISR, self).readConfiguration()
        args = config.OptionConf(self.config['EISR'])
        self.n_layers = int(args['-n_layer'])
        self.num_neighbors = int(args['-num_neighbor'])
        self.tau = float(args['-tau'])
            
    def initModel(self):
        super(EISR, self).initModel()
    
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0] # here we normalize
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix
    
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
    def buildExplicitImplicitMatrix(self):
        social_graph = self.buildSparseRelationMatrix() #m*m
        hetero_graph = self.create_joint_sparse_adj_tensor() #(m+n)*(m+n)
        #social_graph = social_graph.multiply(1.0/social_graph.sum(axis=1).reshape(-1, 1))
        social_graph = self.adj_to_sparse_tensor(social_graph)
        
        return social_graph, hetero_graph
    
    def gumbel_softmax(self, logits, temperature=0.2):
        eps = 1e-10
        u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1) #k*m
        gumbel_noise = -tf.log(-tf.log(u + eps) + eps) #k*m
        y = tf.log(logits + eps) + gumbel_noise #k*m
        return tf.nn.softmax(y / temperature) #k*m
    
    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        original_social_graph, hetero_graph = self.buildExplicitImplicitMatrix()
        original_social_graph = tf.sparse.to_dense(tf.sparse.reorder(original_social_graph)) #prepared for sampleed new graph
        self.implicit_social_graph = tf.placeholder(tf.float32, shape=[self.num_users, self.num_users])
        social_graph = original_social_graph + self.implicit_social_graph
        social_graph = tf.math.divide_no_nan(social_graph, tf.reduce_sum(social_graph, axis=1))
        
        #Training parameter
        for i in ['Social', 'Hetero']:
            self.weights['gating%s'%i] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='gatedWeight%s'%i)
            self.weights['gating_bias%s'%i] = tf.Variable(initializer([1, self.emb_size]), name='gatedWeightBias%s'%i)
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
        self.weights['social2item'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='social2item')
        for k in range(self.n_layers):
            self.weights['vae_encoder%d'%k] = tf.Variable(initializer([self.emb_size, 2*self.emb_size]), name='encoder%d'%k)
            self.weights['vae_decoder%d'%k] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='decoder%d'%k)
        
        def channel_attention(channel_embeddings):
            weights = [] #k*m
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), axis=1)) #m
            score = tf.nn.softmax(tf.transpose(weights), axis=1) #m*k
            mixed_embeddings = 0 #broadcasting
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i]))) #m*d
            return mixed_embeddings,score
        
        #Self-gating
        gating_social = tf.nn.sigmoid(tf.matmul(self.user_embeddings, self.weights['gatingSocial']) + self.weights['gating_biasSocial'])
        user_embeddings_social = tf.multiply(self.user_embeddings, gating_social)
        gating_hetero = tf.nn.sigmoid(tf.matmul(self.user_embeddings, self.weights['gatingHetero']) + self.weights['gating_biasHetero'])
        user_embeddings_hetero = tf.multiply(self.user_embeddings, gating_hetero)
        
        all_user_embeddings_social = [user_embeddings_social]
        all_sample_user_embeddings_social = []
        all_user_embeddings_hetero = [user_embeddings_hetero]
        all_item_embeddings = [self.item_embeddings]
        rec_loss = 0
        kl_loss = 0
        
        #GCN
        for k in range(self.n_layers):
            #Social
            user_embeddings_social = tf.matmul(social_graph, all_user_embeddings_social[k])
            #VAE for social-encoder
            mean_logvar = tf.nn.relu(tf.matmul(user_embeddings_social, self.weights['vae_encoder%d'%k]))
            mean, logvar = tf.split(mean_logvar, [self.emb_size, self.emb_size], axis=1)
            sample_user_embeddings_social = mean + tf.exp(logvar/2) * tf.random.normal(tf.shape(logvar))
            all_user_embeddings_social += [tf.math.l2_normalize(user_embeddings_social, axis=1)]
            all_sample_user_embeddings_social += [tf.math.l2_normalize(sample_user_embeddings_social, axis=1)]
            #VAE for social-decoder
            reconstructed_user_embeddings_social = tf.matmul(sample_user_embeddings_social, self.weights['vae_decoder%d'%k])
            rec_loss += tf.reduce_sum(tf.reduce_mean(tf.math.square(user_embeddings_social - reconstructed_user_embeddings_social), axis=1))
            kl_loss += tf.reduce_sum(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1), axis=0)
            
            #Hetero
            mixed_embeddings_hetero = tf.sparse_tensor_dense_matmul(hetero_graph, tf.concat([all_user_embeddings_hetero[k], all_item_embeddings[k]], axis=0))
            user_embeddings_hetero, item_embeddings = tf.split(mixed_embeddings_hetero, [self.num_users, self.num_items], axis=0)
            all_user_embeddings_hetero += [tf.math.l2_normalize(user_embeddings_hetero, axis=1)]
            all_item_embeddings += [tf.math.l2_normalize(item_embeddings, axis=1)]
            
        user_embeddings_hetero = tf.reduce_sum(all_user_embeddings_hetero, axis=0)
        sample_user_embeddings_social, score = channel_attention(all_sample_user_embeddings_social) #graph layer-wise attention
        item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0)
        
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings = tf.matmul(sample_user_embeddings_social, self.weights['social2item']) + user_embeddings_hetero
        
        #gumbel sample new relations
        self.u_segment = tf.placeholder(tf.int32)
        self.weights['selector'] = tf.Variable(initializer([self.num_neighbors, self.num_users]), name='selector')
        user_features = tf.matmul(user_embeddings_hetero[self.u_segment:self.u_segment+100], user_embeddings_hetero, transpose_b=True)
        def getNewNeighbor(embedding):
            alpha = tf.nn.softmax(tf.multiply(embedding, self.weights['selector']))
            alpha = self.gumbel_softmax(alpha, self.tau)
            multi_hot_vector = tf.reduce_sum(alpha, axis=0)
            return multi_hot_vector
        implicit_social_graph = tf.vectorized_map(fn=lambda em:getNewNeighbor(em), elems=user_features)
        new_old_mask = tf.math.logical_not(original_social_graph[self.u_segment:self.u_segment+100]>0) #m+*m 
        implicit_social_graph = tf.where(new_old_mask, implicit_social_graph*1.0, implicit_social_graph*0.0) #totally new relations 
        
        implicit_social_graph = tf.cast(implicit_social_graph>=tf.expand_dims(tf.math.top_k(implicit_social_graph, self.num_neighbors)[0][:,-1], axis=-1), tf.float32) #m+*m, top-k new neighbors
        padding = tf.zeros(shape=(self.num_users, self.num_users))
        implicit_social_graph = tf.concat([padding[:self.u_segment], implicit_social_graph, padding[self.u_segment+100:]], axis=0) # m*m ∈ [0~1]
        
        #BPR
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, item_embeddings), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y)+10e-8)) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding)) + 0.01 * kl_loss + 0.1 * rec_loss
        
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.no_improve = {i: 0 for i in range(50)}
        imp_graph = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                _, l, self.U, self.V, k_l, r_l, imp_graph = self.sess.run([train, loss, self.final_user_embeddings, self.final_item_embeddings, kl_loss, rec_loss, implicit_social_graph],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.u_segment: u_i, self.implicit_social_graph: imp_graph})
                print('training:', epoch + 1, 'batch', n, 'loss:', l, 'k_loss:', k_l, 'rec_loss:', r_l)
            self.ranking_performance(epoch)
            
            #early stop
            if self.earlyStop == 50:
                break
        
        print(self.no_improve)
        self.U,self.V = self.bestU,self.bestV
        
    def saveModel(self):
        #feed_dict used for loss funtion training, w/o it still can generate final_user/item embeddings
        imp_graph = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings], feed_dict={self.implicit_social_graph: imp_graph})
        
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            #print(u)
            #return self.sess.run(self.test,feed_dict={self.u_idx:u})
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items