from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
from random import choice
import tensorflow as tf
from scipy.sparse import coo_matrix
from math import sqrt
import numpy as np
import os
from util import config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-10
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1) #k*m
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps) #k*m
    y = tf.log(logits + eps) + gumbel_noise #k*m
    return tf.nn.softmax(y / temperature) #k*m

class ESRF(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
        self.n_layers_G = 2 #the number of layers of the alternative neigbhor generation module (generator)

    def readConfiguration(self):
        super(ESRF, self).readConfiguration()
        args = config.OptionConf(self.config['ESRF'])
        self.K = int(args['-K']) #controling the magnitude of adversarial learning
        self.beta = float(args['-beta']) #the number of alternative neighbors
        self.n_layers_D = int(args['-n_layer']) #the number of layers of the recommendation module (discriminator)

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.transpose().tocsr()
        B = S.multiply(S.transpose())
        U = S - B
        C1 = (U.dot(U)).multiply(U.transpose())
        A1 = C1 + C1.transpose()
        C2 = (B.dot(U)).multiply(U.transpose()) + (U.dot(B)).multiply(U.transpose()) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.transpose()
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.transpose()
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(U) + (U.transpose().dot(U)).multiply(U)
        A5 = C5 + C5.transpose()
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.transpose())).multiply(U.transpose()) + (U.transpose().dot(U)).multiply(B)
        A7 = (U.transpose().dot(B)).multiply(U.transpose()) + (B.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(B)
        self.A8 = (Y.dot(Y.transpose())).multiply(B)
        A9 = (Y.dot(Y.transpose())).multiply(U)
        A10  = Y.dot(Y.transpose())
        for i in range(self.num_users):
            A10[i,i]=0
        #user pairs which share less than 5 common purchases are ignored
        A10 = A10.multiply(A10>5)
        #obtain the normalized high-order adjacency
        A = S + A1 + A2 + A3 + A4 + A5 + A6 + A7 + self.A8 + A9 + A10
        A = A.transpose().multiply(1.0/A.sum(axis=1).reshape(1, -1))
        A = A.transpose()
        return A

    def buildMotifGCN(self,adjacency):
        self.relation_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U_r') #if difference between value and mean larger than two stdvar, recreate a new one
        #projection_head = tf.Variable(tf.truncated_normal(shape=[self.emb_size, self.emb_size], stddev=0.005), name='p_h')
        self.userSegment = tf.placeholder(tf.int32) #random number

        #convert sparse matrix to sparse tensor
        adjacency = adjacency.tocoo()
        indices = np.mat([adjacency.row, adjacency.col]).transpose() #return m*2 matrix recording the user-user indices
        #print(adjacency.row,adjacency.col,indices)
        self.A = tf.SparseTensor(indices, adjacency.data.astype(np.float32), adjacency.shape)
        #print(self.A)
        self.adjacency = adjacency.tocsr()
        
        #Motif GCN
        self.g_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_embeddings = [self.relation_embeddings]
        user_embeddings = self.relation_embeddings
        for d in range(self.n_layers_G):
            user_embeddings = tf.sparse_tensor_dense_matmul(self.A, user_embeddings)
            norm_embeddings = tf.math.l2_normalize(user_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        user_embeddings = tf.reduce_mean(all_embeddings,axis=0)
        #user_embeddings = tf.nn.sigmoid(tf.matmul(tf.reduce_sum(all_embeddings, 0), projection_head))
        
        # construct concrete selector layer
        self.g_weights['c_selector'] = tf.Variable(initializer([self.K,self.num_users]), name='c_selector') #k*m
        #to avoid oom, each time we just generate alternative negibhborhood for 100 users
        user_features = tf.matmul(user_embeddings[self.userSegment:self.userSegment+100],user_embeddings,transpose_b=True) #100*d d*m = 100*m
        def getAlternativeNeighborhood(embedding): #embedding: 1*m
            alphaEmbeddings = tf.multiply(embedding, self.g_weights['c_selector']) #multiply broadcasting: k*m
            multi_hot_vector = tf.reduce_sum(self.sampling(alphaEmbeddings), 0) #k*m->1*m
            return multi_hot_vector
        self.alternativeNeighborhood = tf.vectorized_map(fn=lambda em:getAlternativeNeighborhood(em),elems=user_features) #100*m
        paddings = tf.zeros(shape=(self.num_users,self.num_users))
        self.alternativeNeighborhood = tf.concat([paddings[:self.userSegment],self.alternativeNeighborhood,paddings[self.userSegment+100:]],0) # only dense on segment, while others 0
        #decoder
        # reg_loss = 0
        # decoder_weight_sizes = [self.num_users,self.embed_size*4,self.num_users]
        # decoder_layers = 2
        # for d in range(decoder_layers):
        #     self.g_weights['decoder_%d' % d] = tf.Variable(
        #         initializer([decoder_weight_sizes[d], decoder_weight_sizes[d + 1]]), name='decoder_%d' % d)
        #     reg_loss += tf.nn.l2_loss(self.g_weights['decoder_%d' % d])
        #
        # decoderEmbeddings = self.alternativeNeighborhood
        # for d in range(decoder_layers-1):
        #     decoderEmbeddings = tf.matmul(decoderEmbeddings, self.g_weights['decoder_%d' % d])
        #     decoderEmbeddings = tf.nn.relu(decoderEmbeddings)
        # decoderEmbeddings = tf.matmul(decoderEmbeddings, self.g_weights['decoder_%d' % (decoder_layers-1)])
        # decoderEmbeddings = tf.nn.sigmoid(decoderEmbeddings)
        # socialReconstruction = decoderEmbeddings
        #
        # self.A8 = self.A8.todense()
        # self.A8[self.A8 > 1] = 1
        # mask = tf.convert_to_tensor(self.A8)
        # reconstruction = tf.multiply(mask, socialReconstruction-self.A8)
        # self.r_loss = tf.nn.l2_loss(reconstruction) #+0.02*reg_loss
        #
        # #training
        # #self.r_loss = tf.nn.l2_loss(tf.multiply(mask,()))+0.005*reg_loss
        # r_opt = tf.train.AdamOptimizer(0.001)
        # self.r_train = r_opt.minimize(self.r_loss)

    def buildRecGCN(self):
        #parameter init
        self.isSocial = tf.placeholder(tf.int32)
        self.isSocial = tf.cast(self.isSocial, tf.bool)
        self.isAttentive = tf.placeholder(tf.int32)
        self.isAttentive = tf.cast(self.isAttentive, tf.bool)
        self.sampledItems = tf.placeholder(tf.int32)
        self.d_weights = dict()
        
        # decode GCN
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0) #(m+n)*d
        norm_adj = self.create_joint_sparse_adj_tensor() #heterogeneous network norm, from graph....py
        initializer = tf.contrib.layers.xavier_initializer()
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers_D):
            self.d_weights['attention_m1%d' % k] = tf.Variable(
                initializer([self.emb_size, self.emb_size]), name='attention_m1%d' % k)
            self.d_weights['attention_m2%d' % k] = tf.Variable(
                initializer([self.emb_size, self.emb_size]), name='attention_m2%d' % k)
            self.d_weights['attention_v%d' % k] = tf.Variable(
                initializer([1, self.emb_size * 2]), name='attention_v1%d' % k)

        vals, indexes = tf.nn.top_k(self.alternativeNeighborhood, self.K) #get the top-k largest value at every row
        for k in range(self.n_layers_D):
            new_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings) #GCN in user-item pair

            #social attention (applying attention may be a little time-consuming)
            selectedItemEmbeddings = tf.gather(ego_embeddings[self.num_users:],self.sampledItems) #m*d
            indexes = tf.cast(indexes,tf.float32)
            userEmbeddings = tf.matmul(ego_embeddings[:self.num_users],self.d_weights['attention_m1%d' % k]) #m*d
            itemEmbeddings = tf.matmul(selectedItemEmbeddings, self.d_weights['attention_m2%d' % k]) #m*d
            attentionEmbeddings = tf.concat([indexes,userEmbeddings],axis=1) #m*(k+d)
            attentionEmbeddings = tf.concat([attentionEmbeddings, itemEmbeddings], axis=1) #size=m*(k+d+d)
            
            def attention(embedding):
                alternativeNeighors,u_embedding,i_embedding = tf.split(tf.reshape(embedding,[1,self.K+2*self.emb_size]),[self.K,self.emb_size,self.emb_size],axis=1)
                #self.alternativeNeighors = alternativeNeighors[0]
                alternativeNeighors = tf.cast(alternativeNeighors[0],tf.int32) #alternativaNeighbors:(1,k); alternativaNeighbors[0]:(k,) 
                friendsEmbedding = tf.gather(ego_embeddings[:self.num_users],alternativeNeighors) #k*d
                friendsEmbedding = tf.matmul(friendsEmbedding,self.d_weights['attention_m1%d' % k]) #k*d
                i_embedding = tf.reshape(tf.concat([i_embedding] * self.K, 1), [self.K, self.emb_size])
                #friendsEmbedding: k*d; u_embedding: 1*d - broadcasting
                #1*2d k*2d - broadcasting --reduce_sum--> k
                res = tf.reduce_sum(tf.multiply(self.d_weights['attention_v%d' % k],tf.sigmoid(tf.concat([friendsEmbedding+u_embedding, i_embedding],1))), 1)
                weights = tf.nn.softmax(res) #k
                socialEmbedding = tf.matmul(tf.reshape(weights,[1,self.K]),tf.gather(ego_embeddings[:self.num_users],alternativeNeighors))#1*k k*d = 1*d
                return socialEmbedding[0] #1*d -> d
            attentive_socialEmbeddings = tf.vectorized_map(fn=lambda em: attention(em),elems=attentionEmbeddings) #get 1 element(row) of attentinonEmbedding
            
            nonattentive_socialEmbeddings = tf.matmul(self.alternativeNeighborhood, ego_embeddings[:self.num_users]) / self.K #(m*m x m*d)/k, in average take 1/k percent

            def without_attention():
                return nonattentive_socialEmbeddings
            def with_attention():
                return attentive_socialEmbeddings #to use attention, this part should be modified

            def without_social():
                return new_embeddings

            def with_social(embeddings):
                socialEmbeddings = tf.cond(self.isAttentive, lambda: with_attention(), lambda: without_attention())
                return tf.concat([(embeddings[:self.num_users]+socialEmbeddings),embeddings[self.num_users:]],0)

            ego_embeddings = tf.cond(self.isSocial, lambda: with_social(new_embeddings), lambda: without_social())
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.reduce_sum(all_embeddings, 0)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings,[self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)

    def buildGenerator(self):
        y_ui = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)
        currentNeighbors = tf.gather(self.alternativeNeighborhood, self.u_idx)
        friendEmbeddings = tf.matmul(currentNeighbors, self.multi_user_embeddings)/self.K #1*m m*d
        y_vi = tf.reduce_sum(tf.multiply(friendEmbeddings, self.v_embedding), 1)
        self.g_adv_loss = tf.reduce_sum(tf.log(tf.sigmoid(y_ui-y_vi))) #argmin(y_vi-y_ui) is argmax(y_ui-y_vi)
        self.g_loss = self.beta*self.g_adv_loss#+self.r_loss
        opt = tf.train.AdamOptimizer(self.lRate*5)
        self.g_train = opt.minimize(self.g_loss, var_list=[self.g_weights,self.relation_embeddings]) #update g_weight and user_embedding

    def buildDiscriminator(self):
        #Loss for recommendation (BPR)
        y_ui = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)
        y_uj = tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        pairwise_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_ui-y_uj)))
        
        #loss for adversial 
        currentNeighbors = tf.gather(self.alternativeNeighborhood,self.u_idx)
        friendEmbeddings = tf.matmul(currentNeighbors, self.multi_user_embeddings) / self.K
        y_vi = tf.reduce_sum(tf.multiply(friendEmbeddings,self.v_embedding),1)
        adversarial_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_ui-y_vi)))
        
        #Loss for Regularization = 0.03*tf.nn.l2_loss(self.u_embedding - friendEmbeddings)
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) + tf.nn.l2_loss(self.neg_item_embedding))
         
        self.d_loss = pairwise_loss + reg_loss
        self.d_advloss = pairwise_loss + reg_loss + self.beta*adversarial_loss#+s_Regularization
        opt = tf.train.AdamOptimizer(self.lRate)
        self.d_train = opt.minimize(self.d_loss,var_list = [self.user_embeddings,self.item_embeddings])
        self.d_adv_train = opt.minimize(self.d_advloss, var_list=[self.user_embeddings, self.item_embeddings,self.d_weights])
        self.minimax = tf.group(self.d_adv_train, self.g_train)

    def sampling(self,vec):
        vec = tf.nn.softmax(vec) #return k*m
        logits = gumbel_softmax(vec, 0.2) #t=0.2
        return logits

    def initModel(self):
        super(ESRF, self).initModel() #deep-iterative
        self.listed_data = [] #list of user-purchased-item list
        for i in range(self.num_users):
            user = self.data.id2user[i]
            items = list(self.data.trainSet_u[user].keys())
            items = [self.data.item[item] for item in items]
            self.listed_data.append(items)
            #print(len(self.listed_data),len(self.listed_data[0])) 

    def sampleItems(self):
        selectedItems = []
        for i in range(self.num_users):
            item = choice(self.listed_data[i]) #choose one item that the user interested
            selectedItems.append(item)
        #print(len(selectedItems))
        return selectedItems

    def trainModel(self):
        A = self.buildMotifInducedAdjacencyMatrix()
        self.buildMotifGCN(A)
        self.buildRecGCN()
        self.buildGenerator()
        self.buildDiscriminator()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.attentiveTraining = 1
        #pretrain Motif-based GCN

        # for epoch in range(50):
        #     _, l = self.sess.run([self.r_train, self.r_loss])
        #     print 'training:', epoch + 1, 'loss:', l

        #conventional training
        print('pretraining...')
        for epoch in range(self.maxEpoch // 3):
            selectedItems = self.sampleItems() #1*m list
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx= batch
                u_i = np.random.randint(0, self.num_users)
                _, _, self.U, self.V = self.sess.run([self.d_train,self.d_loss,self.multi_user_embeddings, self.multi_item_embeddings],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.userSegment:u_i,
                                                self.isSocial:0,self.isAttentive:self.attentiveTraining,self.sampledItems:selectedItems})
            print(self.foldInfo, 'training:', epoch + 1, 'finished.')
            self.ranking_performance(epoch)
        print('normal training with social relations...')
        for epoch in range(self.maxEpoch // 3):
            selectedItems = self.sampleItems()
            for n, batch in enumerate(self.next_batch_pairwise()):
                u_i = np.random.randint(0, self.num_users)
                user_idx, i_idx, j_idx = batch
                _, _, self.U, self.V = self.sess.run([self.d_train, self.d_loss,self.multi_user_embeddings, self.multi_item_embeddings],
                              feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.userSegment:u_i,
                                         self.isSocial: 1, self.isAttentive: self.attentiveTraining,  self.sampledItems: selectedItems})
            print(self.foldInfo, 'training:', self.maxEpoch//3 + epoch + 1, 'finished.')
            self.ranking_performance(epoch+1*self.maxEpoch//3)
        #adversarial learning without attention
        print('adversarial training with social relations...')
        for epoch in range(self.maxEpoch // 3):
            selectedItems = self.sampleItems()
            for n, batch in enumerate(self.next_batch_pairwise()):
                u_i = np.random.randint(0, self.num_users)
                user_idx, i_idx, j_idx = batch
                self.sess.run([self.minimax], feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.userSegment:u_i,
                                                       self.isSocial:1,self.isAttentive:self.attentiveTraining,self.sampledItems:selectedItems})
            u_i = np.random.randint(0, self.num_users)
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings],
                                            feed_dict={self.userSegment:u_i,self.isSocial: 0,self.isAttentive:0,self.sampledItems:selectedItems})
            
            self.ranking_performance(epoch+2*self.maxEpoch//3)
        self.U, self.V = self.bestU, self.bestV
        
        import pickle 
        with open('ESRF.pkl','wb') as fp:
            pickle.dump(self.U, fp)
        # self.attentiveTraining = 1
        # #adversarial learning with attention
        # for epoch in range(self.maxIter/2):
        #     selectedItems = self.sampleItems()
        #     for n, batch in enumerate(self.next_batch_pairwise()):
        #         user_idx, i_idx, j_idx = batch
        #         self.sess.run(self.minimax, feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isSocial:1,self.isAttentive:self.attentiveTraining,self.sampledItems:selectedItems})
        #
        #     selectedItems = self.sampleItems()
        #     self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings],
        #                                    feed_dict={self.u_idx: [0], self.neg_idx: [0],
        #                                               self.v_idx: [0], self.isSocial: 1,self.isAttentive:1,self.sampledItems:selectedItems})
        #     self.isConverged(epoch + 1+self.maxIter)
            #self.sess.run([self.r_train, self.r_loss])

    def saveModel(self):
        selectedItems = self.sampleItems()
        u_i = np.random.randint(0, self.num_users)
        self.bestU, self.bestV = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings],
                                       feed_dict={self.userSegment:u_i,self.isSocial: 0, self.isAttentive:self.attentiveTraining,self.sampledItems: selectedItems})

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items