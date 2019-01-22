import numpy as np
from sklearn.metrics import roc_auc_score

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0
        # print np.mean(self.mu1 - self.mu0)

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.abs(np.mean(pred_ite) - np.mean(self.true_ite))

    def abs_ate2(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0, y_j):
        # return np.sqrt(np.mean(np.square((1. - 2. * self.t) * (y_j - self.y) - (ypred1 - ypred0))))
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def pehe2(self, ypred1, ypred0):
        # print self.mu1.shape
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        # print ypred.shape
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0, y_j=None):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate2(ypred1, ypred0)
        pehe = self.pehe2(ypred1, ypred0)
        # if y_j is not None:
        #     pehe = self.pehe(ypred1, ypred0, y_j)
        # else:
        #     pehe = None
        return ite, ate, pehe


class EvaluatorTwins(object):
    def __init__(self, y, t, y_cf):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.y1 = self.t * self.y + (1 - self.t) * self.y_cf
        self.y0 = (1 - self.t) * self.y + self.t * self.y_cf
        self.true_ite = self.y1 - self.y0

    def abs_ate(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.abs(np.mean(pred_ite) - np.mean(self.true_ite))

    def abs_ate2(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.y1 - self.y0) - (ypred1 - ypred0))))

    def calc_stats2(self, ypred1, ypred0):
        ate = self.abs_ate2(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ate, pehe
    
    def calc_stats(self, ypred1, ypred0):
        y_cf = (1. - self.t) * ypred1 + self.t * ypred0
        auc = roc_auc_score(self.y_cf, y_cf)
        y_f = (1. - self.t) * ypred0 + self.t * ypred1
        auc_f = roc_auc_score(self.y, y_f)
        return auc, auc_f
    
    def calc_dir_error(self, ypred1, ypred0):
        inds = self.y1!=self.y0
        dir_error = np.mean((np.sign(ypred1[inds]-ypred0[inds]))==(np.sign(self.y1[inds]-self.y0[inds])))
        return dir_error
        


def calc_stats(y, t, y_j, ypred_t1, ypred_t0):
    idx1, idx0 = np.where(t == 1), np.where(t == 0)
    ite1, ite0 = y[idx1] - ypred_t0[idx1], ypred_t1[idx0] - y[idx0]
    eite = (np.sum(np.abs(ite1)) + np.sum(np.abs(ite0))) / y.shape[0]
    eate = np.abs((np.sum(ite1) + np.sum(ite0))) / y.shape[0]
    pehe = np.sqrt(np.mean(np.square((1. - 2. * t) * (y_j - y) - (ypred_t1 - ypred_t0))))
    eate2 = np.abs(np.mean(ite1) - np.mean(ite0))
    return np.array([eite, eate, pehe, eate2])
