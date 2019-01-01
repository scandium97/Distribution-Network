import tensorflow as tf 
import numpy as np 


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def f1_loss(y,eta0,y0):
    return tf.reduce_mean(-tf.sign(y0-y)*eta0 + tf.math.log(1+tf.exp(eta0)))

def fai_loss(y,fai_j,yj,yi):
    #yi == y_j-1
    return tf.reduce_mean(tf.sign(y-yi)*(-tf.sign(yj-y)*fai_j + tf.log(1+tf.exp(fai_j))))

def fai2lambda(fai):
    return sigmoid(fai)

def eta2s(eta):
    return 1 - sigmoid(eta)

def generate_F_list(eta0,fai_list):
    s0 = eta2s(eta0)
    lambda_list = fai2lambda(fai_list).reshape([-1])
    F_list = [1-s0]
    t = s0.copy()
    for lam_i in lambda_list:
        t = t * (1 - lam_i)
        F_list.append(1-t)
    return F_list

def expectation(F_list,anchor):
    #you have to  interpolate for integrating
    tmp_list = zip(F_list_new,anchor_new)
    step = tmp_list[1][1]-tmp_list[0][1]
    value = 0.0
    for i in range(1,len(tmp_list)):
        f, y = tmp_list[i]
        if y>=0: value += (1-f)
        else: value -= f
    return value*step

class distribution_model():
    def __init__(self,isTraining,save_path="./ckpt/tmp/"):
        self.xp = None
        self.yp = None
        self.eta0 = None
        self.loss = None
        self.fai_list = None
        self.output = None
        self.anchor = []
        
        self.isTraining = isTraining
        self.sess = None
        self.train_op = None
        self.step = None
        self.learning_rate = 1e-4
    
    def _build_network(self):
        self.xp = tf.placeholder(dtype=tf.float32,shape=[None,n_seq,n_feature])
        self.yp = tf.placeholder(dtype=tf.float32,shape=[None])
        net = self._generate_feature()
        self.eta0 = tf.dense(net,1,name="eta_dense")
        self.fai_list = []
        self.loss = f1_loss(self.yp,eta0,self.anchor[0])
        
        for i in range(1,len(self.anchor)):
            fai_tmp = tf.dense(net,1,name="fai{}".format(i))
            self.loss += fai_loss(self.yp,fai_tmp,self.anchor[i],self.anchor[i-1])
            self.fai_list.append(fai_tmp)

    def _generate_feature(self):
        net = tf.layers.flatten(self.xp)
        return net

    def _init_session(self):
        return 

    def _init_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate)

        self.step =None
    def fit(self,x,y):
        fetches = [self.loss,self.step]
        loss, __ = sess.run(fetches,{self.xp:x,self.yp:y})
        return loss 

    def validate(self,x,y):
        fetches = [self.loss]
        return sess.run(fetches,{self.xp:x,self.yp:y})[0]

    def predict_distribution(self,x):

        return F_list 

    def predict_value(self,x):
        F_list = self.predict_distribution(x)
        return expectation(F_list,self.anchor)

    def save_model(self,global_step=0):
        return


