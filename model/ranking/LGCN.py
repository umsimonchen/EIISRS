from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util.loss import bpr_loss
from util.config import OptionConf
import numpy as np
import os
import pickle
import fwr13y.d9m.tensorflow as tf_determinism

tf_determinism.enable_determinism()
tf.set_random_seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class LGCN(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LGCN, self).__init__(conf,trainingSet,testSet,fold)
        args = OptionConf(self.config['LGCN'])
        self.n_layers = int(args['-n_layer'])
        self.K = int(args['-K'])
        
    def initModel(self):
        super(LGCN, self).initModel()
        
        # filter
        with open('/root/autodl-tmp/info_A/info_k=%d'%self.K,'rb') as fp:
            info = pickle.load(fp)
        e, v = info[1], info[2]
        v = tf.constant(v, dtype=tf.float32)
        
        self.filters = []
        self.transforms = []
        for k in range(self.n_layers):
            self.filters += [tf.Variable(tf.random_normal([e.shape[0]], mean=0.01, stddev=0.02, dtype=tf.float32, seed=0))]
            self.transforms += [tf.Variable((tf.random_normal([self.emb_size, self.emb_size], mean=0.01, stddev=0.02, dtype=tf.float32, seed=0) + tf.diag(tf.random_normal([self.emb_size], mean=1, stddev=0.001, dtype=tf.float32, seed=0))))]
        
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            new_embeddings = tf.matmul(tf.matmul(v, tf.diag(self.filters[k])), tf.matmul(v, all_embeddings[k], transpose_a=True, transpose_b=False))
            new_embeddings = tf.nn.relu(tf.matmul(new_embeddings, self.transforms[k]))
            all_embeddings += [tf.math.l2_normalize(new_embeddings, axis=1)]    
            
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)
        
        #for lightgcn,this is redundant
        #self.test = tf.reduce_sum(tf.multiply(self.batch_user_emb, self.multi_item_embeddings), 1)

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
                self.batch_neg_item_emb))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(rec_loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l, self.U, self.V = self.sess.run([train, rec_loss, self.multi_user_embeddings, self.multi_item_embeddings],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
            self.ranking_performance(epoch)
    
    def saveModel(self):
        self.bestU, self.bestV = self.U, self.V
        
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
        