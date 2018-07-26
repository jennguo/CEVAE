import numpy as np
from sklearn.model_selection import train_test_split

class IHDP(object):
    def __init__(self, path_data_unformatted, dim_x, replications, bin_feats=()):
        self.path_data_unformatted = path_data_unformatted
        self.replications = replications
        # which features are binary
        self.binfeats = list(bin_feats)
        # which features are continuous
        self.contfeats = [i for i in xrange(dim_x) if i not in self.binfeats]

    def __iter__(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data_unformatted % (i + 1), delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data_unformatted % (i + 1), delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature x[:, 13] is encoded as {1, 2} so subtract 1 to make it {0, 1}
            x[:, 13] -= 1 ### only for the original data!!!
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats

