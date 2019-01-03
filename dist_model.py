import tensorflow as tf 
import numpy as np 
from numba import jit
import random
"""
the first 6 function are inspired by papers -------
<On estimating conditional quantiles and distribution functions>
I'm sorry but I will not explain them very much [for my poor English ......]
"""
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def tf_characteristic(x):
    return tf.nn.relu(tf.sign(x))

def f1_loss(y,eta0,y0):
    lo = tf.reduce_mean( - tf_characteristic(y0-y)*eta0 + tf.log(1 + tf.exp(eta0)))
    return tf.exp(lo)
def fai_loss(y,fai_j,yj,yi):
    #yi == y_j-1
    lo = tf.reduce_mean( tf_characteristic(y-yi)*(-tf_characteristic(yj-y)*fai_j + tf.log(1 + tf.exp(fai_j)) ) )
    return tf.exp(lo)

def fai2lambda(fai):
    return sigmoid(fai)
def eta2s(eta):
    return 1 - sigmoid(eta)

@jit
def generate_F_list(eta0,fai_array):
    """
    eta0: [batch,1] or [batch]
    fai_array: [batch, len(anchor)-1]
    return : ndarray: [batch, len(anchor)] 
    ##lambda_array = fai2lambda(fai_array)
    # tmp_array = log(1-lambda)
    log(Sj) = log(S0) + sum{i}(log(1-lam_i))
    """
    s0 = eta2s(eta0)
    s0 = np.reshape(s0,(s0.shape[0],1))
    tmp_array = -np.log(1+np.exp(fai_array))
    batch, m = tmp_array.shape[0],tmp_array.shape[1]
    for i in range(1,m):
        tmp_array[:,i] += tmp_array[:,i-1]
    tmp_array = np.hstack([np.zeros(shape=[batch,1]),tmp_array])
    ln_s = np.log(s0)+ tmp_array
    F_array = 1 - np.exp(ln_s)
    return F_array

@jit
def expectation(F_array,anchor):
    #
    # 0.0 should be in anchor
    # F_array: shape = [n_batch, len(anchor) ]
    # return : shape = [n_batch] or [n_batch,1]
    # return the expectation of n_batch distributions
    value = np.zeros(shape=(F_array.shape[0],))
    for i in range(1,len(anchor)):
        f, y = F_array[:,i],anchor[i]
        fpv,ypv = F_array[:,i-1],anchor[i-1]
        tmp_s = (f + fpv)/2.0*(y-ypv)
        if y>0: value += (1-tmp_s)
        else: value -= tmp_s
    return value

def clip_gradients(op,losses):
    """
    op: tf.optimizer
    return : global step
    """
    grads = op.compute_gradients(losses)
    capped_grads = [(tf.clip_by_norm(grad, 10.0), var) for grad, var in grads if grad is not None]
    step = op.apply_gradients(capped_grads)
    return step

def _generate_anchor():
    a = [i for i in range(-10,0,1)] +[i for i in range(0,
        11,1)] 
    a.sort()
    a = [float(i)/100.0 for i in a]
    return [-0.1,-0.05,0.0,0.05,0.1]

class distribution_model():
    def __init__(self,isTraining,save_path="./ckpt/tmp/"):
        self.xp = None
        self.yp = None
        self.eta0 = None
        self.loss = None
        self.fai_list = None
        self.output = None
        self.anchor = _generate_anchor()
        self.loss_li = []
        self.isTraining = isTraining
        self.sess = None
        self.train_op = None
        self.step = None
        self.learning_rate = 1e-3

        tf.reset_default_graph()
        self._build_network(n_seq=42,n_feature=6)
        self._init_session()
        if isTraining == True:
            self._init_train_op()
        self._init_variables()
    
    def _build_network(self,n_seq,n_feature):
        """
        use features we generate two elements:
        self.eta0: only 1 value
        self.fai_list: len(list)=len(anchor) - 1
        you might not need loss_li
        """
        self.xp = tf.placeholder(dtype=tf.float32,shape=[None,n_seq,n_feature])
        self.yp = tf.placeholder(dtype=tf.float32,shape=[None,1])
        net = self._generate_feature()
        self.eta0 = tf.layers.dense(net,1,name="eta_dense")
        self.fai_list = []
        self.loss = f1_loss(self.yp,self.eta0,self.anchor[0])
        self.loss_li.append(self.loss)
        for i in range(1,len(self.anchor)):
            fai_tmp = tf.layers.dense(net,1,name="fai{}".format(i))
            self.loss_li.append( fai_loss(self.yp,fai_tmp,self.anchor[i],self.anchor[i-1]) )
            self.loss += self.loss_li[-1]
            self.fai_list.append(fai_tmp)

    def _generate_feature(self):
        """
        the net's input is self.xp
        the output tensor shape can be: [ n_batch, _dim ]
        the user can modify this function to get better features
        """
        net = tf.layers.flatten(self.xp)
        net = tf.layers.dense(net,128,tf.nn.relu)
        net = tf.layers.dense(net,128,tf.nn.relu)
        return net

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        return 

    def _init_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.step = clip_gradients(self.train_op,self.loss)
        return 
    
    def _init_variables(self):
        self.sess.run(tf.global_variables_initializer())
        return 

    def fit(self,x,y):
        total = 0.0
        fetches = [self.loss,self.step]
        total,_s = self.sess.run(fetches,{self.xp:x,self.yp:y})
        return total 

    def validate(self,x,y):
        fetches = [self.loss]
        return self.sess.run(fetches,{self.xp:x,self.yp:y})[0]

    def predict_distribution(self,x):
        """
        note fai_li shape: [len(anchor)-1, n_batch]
        so before we give it to <generate_F_list>, it should be tranposed
        """
        fetches = [self.eta0,self.fai_list]
        eta, fai_li = self.sess.run(fetches,{self.xp:x})
        fai_li = np.reshape(np.array(fai_li),
         (len(self.anchor)-1, eta.shape[0] )   )
        return generate_F_list(eta,fai_li.T) 

    def predict_value(self,x):
        F_list = self.predict_distribution(x)
        # random choose a batch of input data
        # write the predicted distribution for debugging
        r = random.randint(1,40)
        if r == 2:
            np.savetxt("f.npx",F_list)
        return expectation(F_list,self.anchor)

    def save_model(self,global_step=0):
        return





if __name__ == '__main__':
    anchor = _generate_anchor()
    print(anchor)
    np.random.seed(0)
    eta = np.random.uniform(-5.,5.,size=(3,1))
    fai_ay =  np.random.uniform(-5.,5.,size=(3,len(anchor)-1))
    f = generate_F_list(eta,fai_ay)
    print(f)
    print(expectation(f,anchor))
