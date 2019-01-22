from __future__ import absolute_import
from __future__ import division

from parse_data_uriver import Twins
import numpy as np
from numpy import shape, mean, sum, min, max
from evaluation import EvaluatorTwins
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import sem

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# dataset = Twins(treatment='random',noise_bin=0.5)

# RCT equivalent

dataset = Twins(treatment='conf_gest_3', noise_bin=0.333) #hypars here
# Used in main Figure of CEVAE paper, 3 noisy copies of binary GEST10 indicator variables
X, T, yf, y_cf, xtype, rids = dataset.get_data()
# shape(X)
print shape(X)
# len(xtype)

print rids #names of new datas
y0 = yf * (1 - T) + y_cf * T
y1 = yf * T + y_cf * (1 - T)
print mean(y1 - y0)  # ATE
print abs(mean(yf[T == 1]) - mean(yf[T == 0]) - mean(y1 - y0))  # naive estimate minus actual ATE
lr1auc = list()             # LR 1 trn auc's
lr1auc_te = list()          # LR 1 tst auc's
lr2auc = list()             # LR 2 trn auc's
lr2auc_te = list()          # LR 2 tst auc's

lr1auc_fact = list()        #
lr1auc_te_fact = list()
lr2auc_fact = list()
lr2auc_te_fact = list()

lr1dir = list()
lr1dir_te = list()
lr2dir = list()
lr2dir_te = list()

for replic, ((Xtr, Ttr, Ytr, YCFtr), (Xte, Tte, Yte, YCFte)) in enumerate(dataset.get_train_test(n_splits=5)):
    evaluator_tr = EvaluatorTwins(Ytr, Ttr, YCFtr)
    evaluator_te = EvaluatorTwins(Yte, Tte, YCFte)

    lr2_1 = LogisticRegression().fit(Xtr[Ttr.ravel() == 1], Ytr[Ttr.ravel() == 1].ravel())  #
    lr2_0 = LogisticRegression().fit(Xtr[Ttr.ravel() == 0], Ytr[Ttr.ravel() == 0].ravel())
    lr2y0, lr2y1 = lr2_0.predict_proba(Xtr)[:, 1][:, np.newaxis], lr2_1.predict_proba(Xtr)[:, 1][:, np.newaxis]
    lr2y0t, lr2y1t = lr2_0.predict_proba(Xte)[:, 1][:, np.newaxis], lr2_1.predict_proba(Xte)[:, 1][:, np.newaxis]

    lr2auc.append(evaluator_tr.calc_stats(lr2y1, lr2y0)[0])
    lr2auc_te.append(evaluator_te.calc_stats(lr2y1t, lr2y0t)[0])

    lr2auc_fact.append(evaluator_tr.calc_stats(lr2y1, lr2y0)[1])
    lr2auc_te_fact.append(evaluator_te.calc_stats(lr2y1t, lr2y0t)[1])

    lr2dir.append(evaluator_tr.calc_dir_error(lr2y1, lr2y0))
    lr2dir_te.append(evaluator_te.calc_dir_error(lr2y1t, lr2y0t))

    lr1 = LogisticRegression().fit(np.concatenate([Xtr, Ttr], axis=1), Ytr.ravel())
    lr1y0 = lr1.predict_proba(np.concatenate([Xtr, np.zeros_like(Ttr)], axis=1))[:, 1][:, np.newaxis]
    lr1y1 = lr1.predict_proba(np.concatenate([Xtr, np.ones_like(Ttr)], axis=1))[:, 1][:, np.newaxis]
    lr1y0t = lr1.predict_proba(np.concatenate([Xte, np.zeros_like(Tte)], axis=1))[:, 1][:, np.newaxis]
    lr1y1t = lr1.predict_proba(np.concatenate([Xte, np.ones_like(Tte)], axis=1))[:, 1][:, np.newaxis]

    lr1auc.append(evaluator_tr.calc_stats(lr1y1, lr1y0)[0])
    lr1auc_te.append(evaluator_te.calc_stats(lr1y1t, lr1y0t)[0])

    lr1auc_fact.append(evaluator_tr.calc_stats(lr1y1, lr1y0)[1])
    lr1auc_te_fact.append(evaluator_te.calc_stats(lr1y1t, lr1y0t)[1])

    lr1dir.append(evaluator_tr.calc_dir_error(lr1y1, lr1y0))
    lr1dir_te.append(evaluator_te.calc_dir_error(lr1y1t, lr1y0t))

    print 'Replication {}/{}'.format(replic + 1, 5)
    # print 'LR1 train_auc_fact: {:0.3f}, test_auc_fact: {:0.3f}'.format(lr1auc_fact[-1], lr1auc_te_fact[-1])
    # print 'LR2 train_auc_fact: {:0.3f}, test_auc_fact: {:0.3f}'.format(lr2auc_fact[-1], lr2auc_te_fact[-1])
    # print 'LR1 train_auc: {:0.3f}, test_auc: {:0.3f}'.format(lr1auc[-1], lr1auc_te[-1])
    # print 'LR2 train_auc: {:0.3f}, test_auc: {:0.3f}'.format(lr2auc[-1], lr2auc_te[-1])

    # print 'LR1 train_dir: {:0.3f}, test_dir: {:0.3f}'.format(lr1dir[-1], lr1dir_te[-1])
    # print 'LR2 train_dir: {:0.3f}, test_dir: {:0.3f}'.format(lr2dir[-1], lr2dir_te[-1])

print ''
print 'mean LR1 train_auc_fact: {:0.3f}+-{:0.3f}, test_auc_fact: {:0.3f}+-{:0.3f}'.format(mean(lr1auc_fact),
                                                                                          sem(lr1auc_fact),
                                                                                          mean(lr1auc_te_fact),
                                                                                          sem(lr1auc_te_fact))
print 'mean LR2 train_auc_fact: {:0.3f}+-{:0.3f}, test_auc_fact: {:0.3f}+-{:0.3f}'.format(mean(lr2auc_fact),
                                                                                          sem(lr2auc_fact),
                                                                                          mean(lr2auc_te_fact),
                                                                                          sem(lr2auc_te_fact))
print ''
print 'mean LR1 train_auc: {:0.3f}+-{:0.3f}, test_auc: {:0.3f}+-{:0.3f}'.format(mean(lr1auc), sem(lr1auc),
                                                                                mean(lr1auc_te), sem(lr1auc_te))
print 'mean LR2 train_auc: {:0.3f}+-{:0.3f}, test_auc: {:0.3f}+-{:0.3f}'.format(mean(lr2auc), sem(lr2auc),
                                                                                mean(lr2auc_te), sem(lr2auc_te))
print ''
print 'mean LR1 train_dir: {:0.3f}+-{:0.3f}, test_dir: {:0.3f}+-{:0.3f}'.format(mean(lr1dir), sem(lr1dir),
                                                                                mean(lr1dir_te), sem(lr1dir_te))
print 'mean LR2 train_dir: {:0.3f}+-{:0.3f}, test_dir: {:0.3f}+-{:0.3f}'.format(mean(lr2dir), sem(lr2dir),
                                                                                mean(lr2dir_te), sem(lr2dir_te))
