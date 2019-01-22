from __future__ import absolute_import
from __future__ import division

import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Categorical, Poisson, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar

from parse_data_uriver import Twins
import numpy as np
import time
from utils import fc_net, get_y0_y1
from argparse import ArgumentParser
from optimizers import AdamaxOptimizer
from scipy.stats import sem
from evaluation import EvaluatorTwins


parser = ArgumentParser()
parser.add_argument('-ema_every', type=int, default=50)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-use_mean', action='store_true')
parser.add_argument('-super_rate', type=float, default=1.)
parser.add_argument('-splits', default=20)
parser.add_argument('-treat', default='random')
parser.add_argument('-dimh', type=int, default=200)
parser.add_argument('-dimz', type=int, default=20)
parser.add_argument('-nh', type=int, default=3)
parser.add_argument('-noise_bin', type=float, default=0.333)
parser.add_argument('-onlydiff', type=int, default=0)
args = parser.parse_args()

np.random.seed(np.remainder(int(1000000*time.time()), 2**32))
sv = 'M6_twin_runs3/treat_{}_ondf_{}_noise{:0.3f}_dimz{:d}_nh{:d}_dimh{:d}_lr{:0.3f}_shff{:d}'.format(args.treat,args.onlydiff, args.noise_bin, args.dimz, args.nh, args.dimh,  args.lr, np.random.randint(1000000))
f = open(sv, 'wb')
f.write(sv)
f.write('\n')



dataset = Twins(treatment=args.treat,noise_bin=args.noise_bin,onlydiff=np.bool(args.onlydiff))
X, _, _, _, xtype, rid = dataset.get_data()
binfeats = [i for i in xrange(len(xtype)) if xtype[i] == 'bin']
ordfeats = [i for i in xrange(len(xtype)) if xtype[i] == 'ord']
catfeats = [i for i, j in enumerate(xtype) if j in ['cat', 'cyc']]
n_classes = []
for i in xrange(len(xtype)):
    if i not in catfeats:
        n_classes.append(1)
    else:
        n_classes.append(len([elem for elem in np.unique(X[:, i]) if not np.isnan(elem)]))


d = args.dimz  # latent dimension
h, nh = args.dimh, args.nh
lamba = 1e-4  # regularization strength
activation = tf.nn.elu

scores_train = np.zeros((args.splits, 1))
scores_test = np.zeros((args.splits, 1))

scores_train_fact = np.zeros((args.splits, 1))
scores_test_fact = np.zeros((args.splits, 1))

scores_train_dir = np.zeros((args.splits, 1))
scores_test_dir = np.zeros((args.splits, 1))

scores_train_ate = np.zeros((args.splits, 1))
scores_test_ate = np.zeros((args.splits, 1))

scores_train_ate2 = np.zeros((args.splits, 1))
scores_test_ate2 = np.zeros((args.splits, 1))

#iddx = np.sum(X,axis=0)>50
#X = X[:,iddx]
#from itertools import compress
#rid = list(compress(rid, iddx))
#xtype = list(compress(xtype, iddx))



print np.shape(X)
for replic, ((Xtr, Ttr, Ytr, YCFtr), (Xte, Tte, Yte, YCFte)) in enumerate(dataset.get_train_test(n_splits=args.splits)):
    evaluator_tr = EvaluatorTwins(Ytr, Ttr, YCFtr)
    evaluator_te = EvaluatorTwins(Yte, Tte, YCFte)
    

    max_x = np.max(Xtr, axis=0)
    N, dimx = Xtr.shape
    batch_p = (100 * np.bincount(Ttr.ravel()) / float(Ttr.shape[0])).astype(np.int32)
    idx1, idx0 = np.arange(Xtr.shape[0])[Ttr.ravel() == 1], np.arange(Xtr.shape[0])[Ttr.ravel() == 0]
    print 'p(t) = {}'.format(np.bincount(Ttr.ravel()) / float(Ttr.shape[0]))

    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        x_bin = tf.placeholder(tf.float32, [None, len(binfeats)], name='x_bin')
        x_ord = tf.placeholder(tf.float32, [None, len(ordfeats)], name='x_ord')
        x_cats = []
        for i in xrange(len(catfeats)):
            x_cats.append(tf.placeholder(tf.int32, [None,], name='x_cat_{}'.format(i+1)))
        x_phs = [x_bin] + x_cats + [x_ord]

        t_ph = tf.placeholder(tf.float32, [None, 1], name='t_ph')
        t_ph2 = tf.placeholder(tf.float32, [None, 1], name='t_ph2')
        y_ph = tf.placeholder(tf.float32, [None, 1], name='y_ph')

        x_phs2 = [x_ph if len(x_ph.get_shape()) == 2 else tf.expand_dims(x_ph, 1) for x_ph in x_phs]
        x_ph = tf.concat([tf.cast(xx, tf.float32) for xx in x_phs2], 1)
        x_ph = x_ph / max_x

        def feed_dict(x, t, tq, y):
            inp = {}
            inp[x_bin], inp[x_ord] = x[:, 0:len(binfeats)], x[:, (len(binfeats) + len(catfeats)):(len(binfeats) + len(catfeats) + len(ordfeats))]
            inp[t_ph], inp[t_ph2], inp[y_ph] = t, tq, y
            for i in xrange(len(x_cats)):
                inp[x_cats[i]] = x[:, len(binfeats) + i]
            return inp

        tr1, tr0 = np.ones_like(Ttr), np.zeros_like(Ttr)
        f1, f0 = feed_dict(Xtr, tr1, Ttr, Ytr), feed_dict(Xtr, tr0, Ttr, Ytr)

        tr1t, tr0t = np.ones_like(Tte), np.zeros_like(Tte)
        f1t, f0t = feed_dict(Xte, tr1t, Tte, Yte), feed_dict(Xte, tr0t, Tte, Yte)

        # Probability model (subgraph)
        # p(z)
        z = Normal(mu=tf.zeros([tf.shape(x_phs[0])[0], d]), sigma=tf.ones([tf.shape(x_phs[0])[0], d]))

        # p(x|z)
        hx = fc_net(z, (np.max((nh - 1,1))) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
        xs = []
        logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin', lamba=lamba)
        xs.append(Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px'))

        lam = fc_net(hx, [h], [[len(ordfeats), tf.nn.softplus]], 'px_z_ord', lamba=lamba)
        xs.append(Poisson(lam=lam, name='poisson_px', value=tf.zeros_like(lam)))

        for i in xrange(len(catfeats)):
            logits = fc_net(hx, [h], [[n_classes[len(binfeats) + i], None]], 'px{}_z'.format(i + 1), lamba=lamba)
            xs.append(Categorical(logits=logits, dtype=tf.int32, name='categorical_px_{}'.format(i + 1)))

        # p(t|z)
        logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        # inpt = tf.concat([z, t], 1)
        logits2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        logits2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Bernoulli(logits=t * logits2_t1 + (1. - t) * logits2_t0, dtype=tf.float32)

        # Variational model (subgraph)
        # q(t|x)
        logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
        qt = Bernoulli(logits=logits_t, dtype=tf.float32)
        # qt = Bernoulli(logits=tf.zeros((tf.shape(x_ph)[0], 1)), dtype=tf.float32)
        # q(y|x,t)
        hqy = fc_net(x_ph, (np.max((nh - 1,1))) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
        logits_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
        logits_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
        qy = Bernoulli(logits=qt * logits_qy_t1 + (1. - qt) * logits_qy_t0, dtype=tf.float32)
        # q(z|x,t,y)

        inpt2 = tf.concat([x_ph, qy], 1)
        hqz = fc_net(inpt2, (np.max((nh - 1,1))) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba, activation=activation)
        muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba, activation=activation)
        qz = Normal(mu=qt * muq_t1 + (1. - qt) * muq_t0, sigma=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

        inpt2 = tf.concat([x_ph, y_ph], 1)
        hqz = fc_net(inpt2, (np.max((nh - 1,1))) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba, activation=activation)
        muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba, activation=activation)
        qz_train = Normal(mu=t_ph * muq_t1 + (1. - t_ph) * muq_t0, sigma=t_ph * sigmaq_t1 + (1. - t_ph) * sigmaq_t0)

        # Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
        data = {}
        data[xs[0]], data[xs[1]] = x_bin, x_ord
        for i in xrange(len(catfeats)):
            data[xs[2 + i]] = x_cats[i]
        data[y], data[t], data[qy], data[qt] = y_ph, t_ph, y_ph, t_ph

        # sess = ed.get_session()
        tf.set_random_seed(1)
        np.random.seed(1)

        inference = ed.KLqp({z: qz_train}, data=data)
        # optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-6)
        # optimizer = AdamaxOptimizer(learning_rate=lr)
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(args.lr, global_step, 100, 0.97, staircase=True)
        # lr = args.lr
        # if args.opt == 'adamax':
        optimizer = AdamaxOptimizer(learning_rate=lr)
        # else:
        #     optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # train_step = optimizer.minimize(CFR.tot_loss, global_step=global_step)
        inference.initialize(optimizer=optimizer, global_step=global_step, scale={y: args.super_rate})
        # inference.initialize(optimizer=optimizer, scale={y: args.super_rate, qy: args.super_rate})
        print 'initialized model successfully.'

        # posterior predictive for p(y|z,t)
        y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        n_epoch, check_ema = args.epochs, args.ema_every
        n_iter_per_epoch = int(Xtr.shape[0] / 100)
        print 'n_epochs: {}, n_iter_per_epoch: {}'.format(n_epoch, n_iter_per_epoch)
        # idx = np.arange(Xtr.shape[0])
        # idx1 = np.arange(Xtr.shape[0])[Ytr.ravel() == 1]
        # idx0 = np.arange(Xtr.shape[0])[Ytr.ravel() == 0]
        Leval, Lfn = 1, 100 if not args.use_mean else 1

        for epoch in range(n_epoch):
            avg_loss = 0.0

            t0 = time.time()
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
            pbar.start()
            # np.random.shuffle(idx)
            for j in range(n_iter_per_epoch):
                pbar.update(j)
                # batch = np.random.choice(idx, 100)
                batch1 = np.random.choice(idx1, batch_p[1])
                batch0 = np.random.choice(idx0, batch_p[0])
                batch = np.concatenate([batch1, batch0], axis=0)
                x_train, y_train, t_train = Xtr[batch], Ytr[batch], Ttr[batch]
                info_dict = inference.update(feed_dict=feed_dict(x_train, t_train, t_train, y_train))
                avg_loss += info_dict['loss']

            # Print a lower bound to the average marginal likelihood for a datapoint.
            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / 100
            y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=Ytr.shape, L=Leval)
            y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=Yte.shape, L=Leval)
            
            auc = evaluator_tr.calc_stats(y1, y0)[0]
            auc_te = evaluator_te.calc_stats(y1t, y0t)[0]
            
            ate = evaluator_tr.abs_ate(y1,y0)
            ate2 = evaluator_tr.abs_ate2(y1,y0)
            ate_te = evaluator_te.abs_ate(y1t,y0t)
            ate2_te = evaluator_te.abs_ate2(y1t,y0t)
            
            auc_fact = evaluator_tr.calc_stats(y1, y0)[1]
            auc_te_fact = evaluator_te.calc_stats(y1t, y0t)[1]
            
            dir_err = evaluator_tr.calc_dir_error(y1, y0)
            dir_err_te = evaluator_te.calc_dir_error(y1t, y0t)

            string = "Epoch: {}/{}, log p(x) >= {:0.3f}, tr_auc: {:0.3f}, te_auc: {:0.3f}," \
                     "tr_auc_fact: {:0.3f}, te_auc_fact: {:0.3f} , tr_dir: {:0.3f}, te_dir: {:0.3f}, dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, auc, auc_te,auc_fact,auc_te_fact, dir_err,dir_err_te, time.time() - t0)
            # if epoch % args.earl == 0 or epoch == (n_epoch - 1):
            #     logpvalid = sess.run(logpy, feed_dict={x_ph: xvalid, t_ph: tvalid, y_ph: yvalid, t_ph2: tvalid})
            #     if logpvalid >= best_logpvalid:
            #         print 'Improved validation accuracy, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid)
            #         best_logpvalid = logpvalid
            #         saver.save(sess, 'models/m6-jobs')

            print string

        # saver.restore(sess, 'models/m6-jobs')
        y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=Ytr.shape, L=Lfn)
        y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=Yte.shape, L=Lfn)
        auc = evaluator_tr.calc_stats(y1, y0)[0]
        auc_te = evaluator_te.calc_stats(y1t, y0t)[0]
        
        auc_fact = evaluator_tr.calc_stats(y1, y0)[1]
        auc_te_fact = evaluator_te.calc_stats(y1t, y0t)[1]
        
        ate = evaluator_tr.abs_ate(y1,y0)
        ate2 = evaluator_tr.abs_ate2(y1,y0)
        ate_te = evaluator_te.abs_ate(y1t,y0t)
        ate2_te = evaluator_te.abs_ate2(y1t,y0t)
            
        dir_err = evaluator_tr.calc_dir_error(y1, y0)
        dir_err_te = evaluator_te.calc_dir_error(y1t, y0t)
        # print 'Orig', ate, pehe
        # from sklearn.linear_model import LogisticRegression
        # lr1 = LogisticRegression().fit(Xtr[Ttr.ravel() == 1], Ytr[Ttr.ravel() == 1].ravel())
        # lr0 = LogisticRegression().fit(Xtr[Ttr.ravel() == 0], Ytr[Ttr.ravel() == 0].ravel())
        # y0, y1 = lr0.predict(Xtr)[:, np.newaxis], lr1.predict(Xtr)[:, np.newaxis]
        # y0, y1 = np.zeros_like(Ytr), np.zeros_like(Ytr)
        # ate, pehe = evaluator_tr.calc_stats(y1, y0)
        # # print 'Rand', ate, pehe
        # y0, y1 = np.zeros_like(Yte), np.zeros_like(Yte)
        # # y0, y1 = lr0.predict(Xte)[:, np.newaxis], lr1.predict(Xte)[:, np.newaxis]
        # ate_te, pehe_te = evaluator_te.calc_stats(y1, y0)

        print 'Replication {}/{}'.format(replic + 1, args.splits)
        print 'train_auc_fact: {:0.3f}, test_auc_fact: {:0.3f}'.format(auc_fact, auc_te_fact)
        print 'train_auc: {:0.3f}, test_auc: {:0.3f}'.format(auc, auc_te)
        print 'train_dir_err: {:0.3f}, test_dir_err: {:0.3f}'.format(dir_err, dir_err_te)
        

        scores_train[replic, :] = np.array([auc])
        scores_test[replic, :] = np.array([auc_te])
        scores_train_fact[replic, :] = np.array([auc_fact])
        scores_test_fact[replic, :] = np.array([auc_te_fact])
        scores_train_dir[replic, :] = np.array([dir_err])
        scores_test_dir[replic, :] = np.array([dir_err_te])
        
        scores_train_ate[replic,:] = np.array([ate])
        scores_train_ate2[replic,:] = np.array([ate2])
        scores_test_ate[replic,:] = np.array([ate_te])
        scores_test_ate2[replic,:] = np.array([ate2_te])
            
        sess.close()

means_train, sems_train = np.mean(scores_train, axis=0), sem(scores_train, axis=0)
means_test, sems_test = np.mean(scores_test, axis=0), sem(scores_test, axis=0)

means_train_fact, sems_train_fact = np.mean(scores_train_fact, axis=0), sem(scores_train_fact, axis=0)
means_test_fact, sems_test_fact = np.mean(scores_test_fact, axis=0), sem(scores_test_fact, axis=0)

means_train_dir, sems_train_dir = np.mean(scores_train_dir, axis=0), sem(scores_train_dir, axis=0)
means_test_dir, sems_test_dir = np.mean(scores_test_dir, axis=0), sem(scores_test_dir, axis=0)

means_ate_train, sems_ate_train = np.mean(scores_train_ate, axis=0), sem(scores_train_ate, axis=0)
means_ate_test, sems_ate_test = np.mean(scores_test_ate, axis=0), sem(scores_test_ate, axis=0)

means_ate2_train, sems_ate2_train = np.mean(scores_train_ate2, axis=0), sem(scores_train_ate2, axis=0)
means_ate2_test, sems_ate2_test = np.mean(scores_test_ate2, axis=0), sem(scores_test_ate2, axis=0)


print 'M6 model final statistics'


strr = 'train_auc_fact: {:0.3f}+-{:0.3f}, test_auc_fact: {:0.3f}+-{:0.3f}'.format(means_train_fact[0], sems_train_fact[0], means_test_fact[0],sems_test_fact[0])
print(''.join(strr))
f.write(''.join(strr))
f.write('\n')

strr = 'train_auc: {:0.3f}+-{:0.3f}, test_auc: {:0.3f}+-{:0.3f}'.format(means_train[0], sems_train[0], means_test[0], sems_test[0])
print(''.join(strr))
f.write(''.join(strr))
f.write('\n')

strr = 'train_dir_err: {:0.3f}+-{:0.3f}, test_dir_err: {:0.3f}+-{:0.3f}'.format(means_train_dir[0], sems_train_dir[0], means_test_dir[0], sems_test_dir[0])
print(''.join(strr))
f.write(''.join(strr))
f.write('\n')

strr = 'train_ate: {:0.3f}+-{:0.3f}, test_ate: {:0.3f}+-{:0.3f}'.format(means_ate_train[0], sems_ate_train[0], means_ate_test[0],sems_ate_test[0])
print(''.join(strr))
f.write(''.join(strr))
f.write('\n')

strr = 'train_ate2: {:0.3f}+-{:0.3f}, test_ate2: {:0.3f}+-{:0.3f}'.format(means_ate2_train[0], sems_ate2_train[0], means_ate2_test[0],sems_ate2_test[0])
print(''.join(strr))
f.write(''.join(strr))
f.write('\n')

f.close()

