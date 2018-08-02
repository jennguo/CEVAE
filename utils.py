import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
from tensorflow.contrib.layers.python.layers import initializers

SQRT_CONST = 1e-10

def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False)):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in xrange(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print
    return y0, y1


# copied from cfrnet
def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):


    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
    Mt = tf.concat([M,row],0)
    Mt = tf.concat([Mt,col],1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))],0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

# copied from cfrnet
# """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
def pdist2sq(X,Y):
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keepdims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

# copied from cfrnet
# ''' Numerically safe version of TensorFlow sqrt '''
def safe_sqrt(x, lbound=SQRT_CONST):
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

