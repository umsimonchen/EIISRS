# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:10:42 2022

@author: simon
"""

from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
import os
from util import config
import pickle5 as pickle
import random
from tqdm import tqdm 
import numpy as np
from time import strftime,localtime,time
from util.io import FileIO
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#For general comparison. We do not include the user/item features extracted from text/images

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(0)
tf.reset_default_graph()
tf.compat.v1.set_random_seed(0)
np.random.seed(0)

class Comirec(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(Comirec, self).readConfiguration()
        args = config.OptionConf(self.config['Comirec'])
        self.len_seq = int(args['-len_seq'])
        self.num_interests = int(args['-num_interests'])
        self.neg_num = int(args['-neg_num'])
        self.mask_ornot = int(args['-interest_mask'])
            
    def initModel(self):
        super(Comirec, self).initModel()
        #read file
        with open('C:/Users/simon/crp_small_matrix_train_data.pkl','rb') as f:
            train_data = pickle.load(f) 
        with open('C:/Users/simon/crp_small_matrix_val_data.pkl','rb') as f:
            val_data = pickle.load(f)
        with open('C:/Users/simon/crp_small_matrix_test_data.pkl','rb') as f:
            test_data = pickle.load(f)
        
        #only take train item set into account because it's the whole set
        self.item_list = list(set(train_data['click_seq'].reshape(-1)).union(set(train_data['pos_item']), set(train_data['neg_item'].reshape(-1))))
        self.user_list = list(set(train_data['user']))
        self.num_users = len(self.user_list)
        self.num_items = len(self.item_list)
        
        #item to id dictionary
        item2id = {}
        for i in range(len(self.item_list)):
            item2id[self.item_list[i]] = i
        user2id = {}
        for i in range(len(self.user_list)):
            user2id[self.user_list[i]] = i
        
        #generate training set
        self.training_set = []
        print("Loading training set...\n")
        #for i in tqdm(range(len(train_data['click_seq']))):
        for i in tqdm(range(10000)):
            sample = []
            click_seq = [item2id[elem] for elem in train_data['click_seq'][i]]
            sample.append(click_seq)
            neg_item = [item2id[elem] for elem in train_data['neg_item'][i]]
            sample.append(neg_item)
            sample.append(item2id[train_data['pos_item'][i]])
            sample.append(user2id[train_data['user'][i]])
            self.training_set.append(sample)
        self.source = self.training_set[0][0]
        
        #generate validation set
        print("Loading validation set...\n")
        self.val_sequence_idx, self.val_neg_idx, self.val_pos_idx, self.val_user_idx = [], [], [], []
        for i in tqdm(range(len(val_data['click_seq']))):
            click_seq = [item2id[elem] for elem in val_data['click_seq'][i]]
            self.val_sequence_idx.append(click_seq)
            neg_item = [item2id[elem] for elem in val_data['neg_item'][i]]
            self.val_neg_idx.append(neg_item)
            self.val_pos_idx.append(item2id[val_data['pos_item'][i]])
            self.val_user_idx.append(item2id[val_data['user'][i]])
        
        #generate test set
        print("Loading test set...\n")
        self.test_sequence_idx, self.test_neg_idx, self.test_pos_idx, self.test_user_idx = [], [], [], []
        for i in tqdm(range(len(test_data['click_seq']))):
            click_seq = [item2id[elem] for elem in test_data['click_seq'][i]]
            self.test_sequence_idx.append(click_seq)
            neg_item = [item2id[elem] for elem in test_data['neg_item'][i]]
            self.test_neg_idx.append(neg_item)
            self.test_pos_idx.append(item2id[test_data['pos_item'][i]])
            self.test_user_idx.append(item2id[test_data['user'][i]])
        
    def next_batch_pairwise(self):
        self.train_size = len(self.training_set)
        random.shuffle(self.training_set)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                end = batch_id + self.batch_size
            else:
                end = len(self.training_set)

            sequence_idx, neg_idx, pos_idx, user_idx = [], [], [], []
            for i, sample in enumerate(self.training_set[batch_id:end]):
                sequence_idx.append(sample[0])
                neg_idx.append(sample[1])
                pos_idx.append(sample[2])
                user_idx.append(sample[3])
            
            batch_id += self.batch_size
            yield sequence_idx, neg_idx, pos_idx, user_idx
            
    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.sequence_idx = tf.placeholder(tf.int32, [None, self.len_seq], name="sequence_idx")
        self.neg_idx = tf.placeholder(tf.int32, [None, None], name="neg_idx")
        self.pos_idx = tf.placeholder(tf.int32, [None], name="pos_idx")
        self.user_idx = tf.placeholder(tf.int32, [None], name="user_idx")
        
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') # n*d
        self.position_embeddings = tf.Variable(tf.truncated_normal(shape=[self.len_seq, self.emb_size], stddev=0.005))
        self.users_k = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.num_interests], stddev=0.005, mean=1, name='interest_mask'))
        self.weights['layer_1'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_1')
        self.weights['layer_2'] = tf.Variable(initializer([self.emb_size, self.num_interests]), name='layer_2')
        
        self.sequence_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.sequence_idx)
        self.sequence_embeddings += tf.tile([self.position_embeddings], [tf.shape(self.sequence_embeddings)[0], 1, 1]) #[b,seq_len,d]
        
        #self-attention
        att = tf.matmul(self.sequence_embeddings, self.weights['layer_1']) #[b,seq_len,d]
        att = tf.nn.tanh(att)
        att = tf.matmul(att, self.weights['layer_2']) #[b,seq_len,num_interests]
        att = tf.nn.softmax(att, axis=1)
        att = tf.transpose(att, [0, 2, 1]) #[b,num_interests,seq_len]
        self.interests_embeddings = tf.matmul(att, self.sequence_embeddings) #[b,num_interests,d]
        
        #find the label of each item in a sequence
        a = tf.expand_dims(tf.math.l2_normalize(self.sequence_embeddings, axis=-1), axis=2)
        b = tf.expand_dims(tf.math.l2_normalize(self.interests_embeddings, axis=-1), axis=1)
        distance = tf.reduce_sum(tf.multiply(a, b), axis=-1) #[b,seq_len,num_interests]
        distance = tf.argsort(distance, direction='DESCENDING')
        
        #training loss
        self.neg_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_idx)
        self.pos_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_idx)
        
        self.neg_score = tf.reduce_max(
            tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.interests_embeddings, axis=1), tf.expand_dims(self.neg_embeddings, axis=2))
            , axis=-1)
        , axis=-1) #[b,1,num_interest,d]*[b,num_neg,1,d]->[b,num_neg,num_interest,d]->[b,num_neg,num_interest]->[b,num_neg]
        self.pos_score = tf.reduce_max(
            tf.reduce_sum(
                tf.multiply(self.interests_embeddings, tf.expand_dims(self.pos_embeddings, axis=1))
            , axis=-1)
        , axis=-1, keepdims=True) #[b,1]
        
        #update
        logits = tf.nn.softmax(tf.concat([self.pos_score, self.neg_score], axis=-1))
        loss = tf.reduce_sum(-tf.math.log(logits[:,0]))
        
        #training epoch
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.bestP ={'epoch':0, 'hr':0.0, 'mrr':0.0, 'ndcg':0.0}
        tmp = []
        same_sequence_prob = []
        same_sequence = []
        check = []
        for epoch in range(100):
            for n, batch in enumerate(self.next_batch_pairwise()):
                sequence_idx, neg_idx, pos_idx, user_idx = batch
                _, l, k, dis, interest_prob = self.sess.run([train, loss, self.users_k, distance, att],
                                     feed_dict={self.sequence_idx: sequence_idx, self.neg_idx: neg_idx, self.pos_idx: pos_idx, self.user_idx: user_idx})
                print('training:', epoch+1, 'batch', n+1, 'loss:', l)
                if len(tmp) != 0:
                    print(n,(tmp==k).all())
                tmp = k

                for seq in range(len(sequence_idx)):
                    if sequence_idx[seq] == self.source:
                        same_sequence_prob.append(interest_prob[seq])
                        same_sequence.append(dis[seq])
                        check.append(sequence_idx[seq])
                
                # if epoch == 25:
                #     for seq in dis:
                #         same_sequence.append(seq)
            
            self.logits = self.sess.run([logits], feed_dict={self.sequence_idx: self.val_sequence_idx, self.neg_idx: self.val_neg_idx, self.pos_idx: self.val_pos_idx, self.user_idx: self.val_user_idx})
            self.evaluation(str(epoch+1), 1)
            
            if epoch - int(self.bestP['epoch']) > 50:
                print('Early stop!!!')
                break
        with open('mask.pkl', 'wb') as fp:
            pickle.dump([same_sequence, check],fp)
        
        # testset evaluation
        self.logits = self.sess.run([logits], feed_dict={self.sequence_idx: self.test_sequence_idx, self.neg_idx: self.test_neg_idx, self.pos_idx: self.test_pos_idx, self.user_idx: self.test_user_idx})
        self.eval_result = self.evaluation('Test', 10)
            
    def saveModel(self):
        pass

    def evalRanking(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.log.add('###Evaluation Results###')
        self.log.add(self.eval_result)
        FileIO.writeFile(outDir, fileName, self.eval_result)
    
    def evaluation(self, epoch, k):
        logits = -self.logits[0]
        rank = logits.argsort().argsort()[:,0]
        
        #metric measure
        hr = 0.0
        mrr = 0.0
        ndcg = 0.0
        for r in rank:
            if r<k:
                #hit rate
                hr += 1
                #mean reciprocal rank
                mrr += 1 / (r+1)
                #normalize discounted cumulative gain
                ndcg += 1 / np.log2(r+2)
        hr /= len(rank)
        mrr /= len(rank)
        ndcg /= len(rank)
        
        #update best performance
        flag = 0
        if hr>self.bestP['hr']: flag+=1
        if mrr>self.bestP['mrr']: flag+=1
        if ndcg>self.bestP['ndcg']: flag+=1
        if flag>1: 
            self.bestP['hr'] = hr
            self.bestP['mrr'] = mrr
            self.bestP['ndcg'] = ndcg
            self.bestP['epoch'] = epoch
        
        #output message
        print('-'*120)
        print('Quick Ranking Performance '+self.foldInfo+' (Top-'+str(k)+'Item Recommendation)')
        
        cp = ''
        cp += 'Hit Rate'+':'+str(hr)+' | '
        cp += 'Mean Reciprocal Rank' + ':' + str(mrr) + ' | '
        cp += 'NDCG' + ':' + str(ndcg)
        print('*Current Performance*')
        print('Epoch:',epoch+',',cp)
    
        if epoch != 'Test':
            bp = ''
            bp += 'Hit Rate'+':'+str(self.bestP['hr'])+' | '
            bp += 'Mean Reciprocal Rank' + ':' + str(self.bestP['mrr']) + ' | '
            bp += 'NDCG' + ':' + str(self.bestP['ndcg'])
            print('*Best Performance* ')
            print('Epoch:',self.bestP['epoch']+',',bp)
            print('-'*120)
            
        return cp
        
        
        
        
        
        
        
        