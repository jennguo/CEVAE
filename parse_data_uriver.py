import numpy as np
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist
#from keras.utils.np_utils import to_categorical


def convert(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

class Twins(object):
    def __init__(self, path_data='datasets/Twins/', treatment='random',noise_bin=0.333,onlydiff=False):
        self.path_data = path_data
        self.treatment = treatment
        self.noise_bin = noise_bin
        self.onlydiff = onlydiff
    def load_data(self):
        import csv, re
        X = []
        with open(self.path_data + 'twin_pairs_X_3years_samesex.csv', 'rb') as handle:
            for i, row in enumerate(csv.reader(handle, delimiter=',')):
                if i == 0:
                    rid = row
                else:
                    X.append(map(convert, row))
        rid = rid[2:]
        X = np.array(X)[:, 2:]
        T = np.loadtxt(self.path_data + 'twin_pairs_T_3years_samesex.csv', delimiter=',', skiprows=1, usecols=[1, 2])
        Y = np.loadtxt(self.path_data + 'twin_pairs_Y_3years_samesex.csv', delimiter=',', skiprows=1, usecols=[1, 2])
        
        if self.onlydiff == True:
            diinds = Y[:,0]!=Y[:,1]
            X = X[diinds,:]
            Y = Y[diinds,:]
            T = T[diinds,:]
        with open(self.path_data + 'covar_type.txt', 'rb') as handle:
            types = eval(handle.read())
        xtype = [types[re.sub("_[0-1]|_reg", "", r)] for r in rid]
        toremove = [i for i in xrange(len(rid)) if rid[i] == 'brstate' or rid[i] == 'stoccfipb' or rid[i] == 'mplbir']
        toremove += [i for i, j in enumerate(xtype) if j == 'index do not use']
        X, xtype, rids = np.delete(X, toremove, 1), [j for i, j in enumerate(xtype) if i not in toremove], [j for i, j in enumerate(rid) if i not in toremove]
        idx = np.argsort(xtype)
        X, xtype, rids = X[:, idx], [xtype[i] for i in idx], [rids[i] for i in idx]
        # print [(i, rids[i]) for i in xrange(len(rids))]
        # raw_input()
        return X, T, Y, xtype, rids

    def treatment_assignment(self, X,rids):
        np.random.seed(1)
        if self.treatment == 'random':
            t = np.random.binomial(1, 0.5, size=(X.shape[0],))
        elif self.treatment == 'logistic':
            w = .3 + np.random.randn(X.shape[1])
            # remove NaN and rescale input features
            Xc = np.copy(X)
            Xc[np.isnan(Xc)] = 0
            Xc /= np.max(Xc, axis=0)
            p = 1. / (1. + np.exp(- np.dot(Xc, w[:, np.newaxis]).ravel()))
            t = np.random.binomial(1, p)
        elif self.treatment == 'conf_gest' or 'conf_gest_3' or 'X_to_bin' or 'X_to_bin_3' or 'conf_gest_3_mult' or 'conf_gest_2_mult':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])            
            means = np.ones((X.shape[1],)) * .0
            means[tohide] = 5.
            w = means + np.random.randn(X.shape[1])*0.1
            Xc = np.copy(X)
            Xc[np.isnan(Xc)] = 0    
            #Xc /= np.max(Xc, axis=0)
            mm = np.max(Xc, axis=0)        
            mm = np.max((mm,np.ones(np.shape(mm))),axis=0)            
            Xc /= mm
            Xc[:,tohide] = Xc[:,tohide]-0.1
            p = 1. / (1. + np.exp(- np.dot(Xc, w[:, np.newaxis]).ravel()))
            t = np.random.binomial(1, p)
        elif self.treatment == 'conf_gest_useless':            
            to_use_names = ['gestatcat' + str(i+1) for i in range(10)]
            to_use_inds = np.array([rids.index(ii) for ii in to_use_names])            
            means = np.ones((X.shape[1],)) * .0
            means[to_use_inds] = [-2,-1,0,1,2,2,2,2,2,2]
            w = means + np.random.randn(X.shape[1])*0.1
            Xc = np.copy(X)
            Xc[np.isnan(Xc)] = 0            
            Xc /= np.max(Xc, axis=0)
            p = 1. / (1. + np.exp(- np.dot(Xc, w[:, np.newaxis]).ravel()))
            t = np.random.binomial(1, p)
        elif self.treatment == 'conf':            
            tohide_names = ['crace','frace','orfath','mrace','ormoth']
            tohide = np.array([rids.index(ii) for ii in tohide_names])            
            means = np.ones((X.shape[1],)) * .2
            means[tohide] = 3.
            w = means + np.random.randn(X.shape[1])
            Xc = np.copy(X)
            Xc[np.isnan(Xc)] = 0
            Xc /= np.max(Xc, axis=0)
            p = 1. / (1. + np.exp(- np.dot(Xc, w[:, np.newaxis]).ravel()))
            t = np.random.binomial(1, p)
        elif self.treatment == 'conf+':
            tohide = np.array([25, 29, 33, 34, 37, 38, 39, 43])
            means = np.ones((X.shape[1],)) * .05
            means[tohide] = 2.
            w = means + np.random.randn(X.shape[1])
            Xc = np.copy(X)
            Xc[np.isnan(Xc)] = 0
            Xc /= np.max(Xc, axis=0)
            p = 1. / (1. + np.exp(- np.dot(Xc, w[:, np.newaxis]).ravel()))
            t = np.random.binomial(1, p)
        else:
            raise Exception()
        return t

    def get_data(self):
        # TODO: Poisson variables add one extra covariate denoting whether the variable is NaN
        # TODO: t \sim logistic(Wx) or t \sim logistic(Wx + random(z)) or any other artificial hidden confounder
        # TODO: Add mean weight as a feature, perhaps as quantiles / binning
        from sklearn.preprocessing import Imputer
        X, T, Y, xtype, rids = self.load_data()
        idx_tleq = T[:, 1] < 2000
        X, T, Y = X[idx_tleq], T[idx_tleq], Y[idx_tleq]
        # w_add = np.mean(T, axis=1)[:, np.newaxis]
        # w_add = (w_add - np.mean(w_add)) / np.std(w_add)
        
        #remove rare features
        

        
        np.random.seed(1)
        # treatment assignment
        # t = self.treatment_assignment(np.concatenate([X, w_add], axis=1))
        t = self.treatment_assignment(X,rids)
        # factual measurement
        y = Y[np.arange(X.shape[0]), t]
        # counterfactual
        y_cf = Y[np.arange(X.shape[0]), 1 - t]
        # remove birth order measurements (since they are affected by t)
        bords = [rids.index('bord_0'), rids.index('bord_1')]
        Xord0, Xord1 = X[:, bords[0]], X[:, bords[1]]
        rids = [rids[i] for i in xrange(len(rids)) if i not in bords]
        xtype = [xtype[i] for i in xrange(len(xtype)) if i not in bords]
        rids.append('bord'); xtype.append('bin')

        X = np.delete(X, bords, 1)
        # add to X the birth order feature according to the twin chosen
        Xord = t * Xord1 + (1 - t) * Xord0
        X = np.concatenate([X, Xord[:, np.newaxis]], axis=1)
        if self.treatment == 'conf':            
            tohide_names = ['crace','frace','orfath','mrace','ormoth']
            tohide = np.array([rids.index(ii) for ii in tohide_names])            
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            # print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'conf_gest':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            # print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'conf_gest_3':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            gestat_onehot = enc.fit_transform(X[:,tohide[0]:tohide[0]+1]).toarray()
            gestat_onehot = np.concatenate((gestat_onehot,gestat_onehot,gestat_onehot),axis=1)
            gestat_onehot = np.logical_xor(gestat_onehot, np.random.binomial(1,p=self.noise_bin,size=np.shape(gestat_onehot)))
            
            X = np.concatenate((X, gestat_onehot),axis=1)
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            xtype = xtype + 30*['bin']
            rids = rids+3*['gestatcat' + str(i+1) for i in range(10)]
            #print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'conf_gest_3_mult':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            gestat_onehot = enc.fit_transform(X[:,tohide[0]:tohide[0]+1]).toarray()
            gestat_onehot = np.concatenate((gestat_onehot,gestat_onehot,gestat_onehot),axis=1)
            #gestat_onehot = np.logical_xor(gestat_onehot, np.random.binomial(1,p=self.noise_bin,size=np.shape(gestat_onehot)))
            gestat_onehot = gestat_onehot*np.random.binomial(1,p=1-2*self.noise_bin,size=np.shape(gestat_onehot))
            X = np.concatenate((X, gestat_onehot),axis=1)
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            xtype = xtype + 30*['bin']
            rids = rids+3*['gestatcat' + str(i+1) for i in range(10)]
            #print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'conf_gest_2_mult':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            gestat_onehot = enc.fit_transform(X[:,tohide[0]:tohide[0]+1]).toarray()
            gestat_onehot = np.concatenate((gestat_onehot,gestat_onehot),axis=1)
            #gestat_onehot = np.logical_xor(gestat_onehot, np.random.binomial(1,p=self.noise_bin,size=np.shape(gestat_onehot)))
            gestat_onehot = gestat_onehot*np.random.binomial(1,p=1-2*self.noise_bin,size=np.shape(gestat_onehot))
            X = np.concatenate((X, gestat_onehot),axis=1)
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            xtype = xtype + 20*['bin']
            rids = rids+2*['gestatcat' + str(i+1) for i in range(10)]
            #print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'X_to_bin':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            X = np.delete(X, tohide, 1)            
            nan_prob = np.sum(np.isnan(X), axis=0) / (1. * X.shape[0])
            for i, elem in enumerate(nan_prob):
                # for those that have a lot of nans use an extra nan category for the feature
                if elem > 0.0:
                    unq_values = [k for k in np.unique(X[:, i]).tolist() if not np.isnan(k)]
                    unq_values += [np.max(unq_values) + 1]
                    if xtype[i] == 'bin':
                        xtype[i] = 'cat'
                    X[:, i][np.isnan(X[:, i])] = unq_values[-1]
            
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            Xbin = enc.fit_transform(X).toarray()            
            ss = sum(Xbin,0)>=100
            X = Xbin[:,ss]            
            xtype = np.shape(X)[1]*['bin']
            rids = np.shape(X)[1]*['unk']
            #xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            #rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            #xtype = xtype + 30*['bin']
            #rids = rids+3*['gestatcat' + str(i+1) for i in range(10)]
            #print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'X_to_bin_3':
            tohide_names = ['gestat10']
            tohide = np.array([rids.index(ii) for ii in tohide_names])
            X = np.delete(X, tohide, 1)            
            nan_prob = np.sum(np.isnan(X), axis=0) / (1. * X.shape[0])
            for i, elem in enumerate(nan_prob):
                # for those that have a lot of nans use an extra nan category for the feature
                if elem > 0.0:
                    unq_values = [k for k in np.unique(X[:, i]).tolist() if not np.isnan(k)]
                    unq_values += [np.max(unq_values) + 1]
                    if xtype[i] == 'bin':
                        xtype[i] = 'cat'
                    X[:, i][np.isnan(X[:, i])] = unq_values[-1]
            
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder()
            Xbin = enc.fit_transform(X).toarray()            
            ss = sum(Xbin,0)>=50
            Xbin = Xbin[:,ss]            
            Xbin = np.concatenate((Xbin,Xbin,Xbin),axis=1)
            Xbin = np.logical_xor(Xbin, np.random.binomial(1,p=self.noise_bin,size=np.shape(Xbin)))
            X = Xbin
            xtype = np.shape(X)[1]*['bin']
            rids = np.shape(X)[1]*['unk']
            #xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            #rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            #xtype = xtype + 30*['bin']
            #rids = rids+3*['gestatcat' + str(i+1) for i in range(10)]
            #print zip(xtype, rids)
            # raw_input()
        elif self.treatment == 'conf+':
            tohide = np.array([23, 27, 31, 32, 35, 36, 37, 41])
            X = np.delete(X, tohide, 1)
            # print zip(xtype, rids)
            # print [(xtype[k], rids[k]) for k in tohide.tolist()]
            xtype = [xtype[k] for k in xrange(len(xtype)) if k not in tohide.tolist()]
            rids = [rids[k] for k in xrange(len(rids)) if k not in tohide.tolist()]
            # print zip(xtype, rids), 'p(t) = {}'.format(np.bincount(t.ravel()) / float(t.shape[0]))
            # raw_input()

        nan_prob = np.sum(np.isnan(X), axis=0) / (1. * X.shape[0])
        for i, elem in enumerate(nan_prob):
            # for those that have a lot of nans use an extra nan category for the feature
            if elem > 0.2:
                unq_values = [k for k in np.unique(X[:, i]).tolist() if not np.isnan(k)]
                unq_values += [np.max(unq_values) + 1]
                if xtype[i] == 'bin':
                    xtype[i] = 'cat'
                X[:, i][np.isnan(X[:, i])] = unq_values[-1]

        # sort features according to the type
        idx = np.argsort(xtype)
        X, xtype, rids = X[:, idx], [xtype[i] for i in idx], [rids[i] for i in idx]
        # impute remaining nan features
        Xt0 = Imputer(strategy='most_frequent').fit_transform(X[t == 0])
        Xt1 = Imputer(strategy='most_frequent').fit_transform(X[t == 1])
        X[t == 0] = Xt0
        X[t == 1] = Xt1
        for i in xrange(X.shape[1]):
            # hack to make sure that categories start from zero
            if xtype[i] in ['cat', 'cyc']:
                unq = np.unique(X[:, i]).tolist()
                for k, elem in enumerate(unq):
                    X[:, i][X[:, i] == elem] = k

        for i in xrange(X.shape[1]):
            if 2 in np.unique(X[:, i]) and (xtype[i] == 'bin'):
                X[:, i] -= 1
            # print np.unique(X[:, i]), xtype[i], rids[i]
            # raw_input()

        # X = np.concatenate([X, w_add], axis=1)
        # xtype.append('cont'); rids.append('mean_t_weight')
        if self.onlydiff==True:
            iddx = np.sum(X,axis=0)>10
            X = X[:,iddx]
            from itertools import compress
            rids = list(compress(rids, iddx))
            xtype = list(compress(xtype, iddx))
            
        return X, t[:, np.newaxis], y[:, np.newaxis], y_cf[:, np.newaxis], xtype, rids

    def get_train_test(self, test_size=0.2, n_splits=10):
        X, t, y, y_cf, _, _ = self.get_data()
        for i in xrange(n_splits):
            Xtr, Xte, ttr, tte, ytr, yte, ycf_tr, ycf_te = train_test_split(X, t, y, y_cf, test_size=test_size,
                                                                            random_state=i + 1, stratify=y)
            yield (Xtr, ttr, ytr, ycf_tr), (Xte, tte, yte, ycf_te)



class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=1000):
        self.path_data = path_data
        self.replications = replications
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in xrange(25) if i not in self.binfeats]

    def __iter__(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this feature is from 1 to 2
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class IHDP3(object):
    def __init__(self, path_data="datasets/IHDP/csv", noise_cont=0.2, noise_bin=0.2, replications=1000):
        self.path_data = path_data
        self.replications = replications
        self.noise_cont = noise_cont
        self.noise_bin = noise_bin
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in xrange(25) if i not in self.binfeats]

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this feature is from 1 to 2
            x[:, 13] -= 1
            x1, x2, x3 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
            # view1
            x1[:, self.contfeats] = x[:, self.contfeats] + self.noise_cont * np.random.randn(x.shape[0], len(self.contfeats))
            x1[:, self.binfeats] = x[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(x.shape[0], len(self.binfeats)))
            # view2
            x2[:, self.contfeats] = x[:, self.contfeats] + self.noise_cont * np.random.randn(x.shape[0], len(self.contfeats))
            x2[:, self.binfeats] = x[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(x.shape[0], len(self.binfeats)))
            # view3
            x3[:, self.contfeats] = x[:, self.contfeats] + self.noise_cont * np.random.randn(x.shape[0], len(self.contfeats))
            x3[:, self.binfeats] = x[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(x.shape[0], len(self.binfeats)))

            # noisy view of x
            x = np.concatenate([x1[:, self.contfeats], x2[:, self.contfeats], x3[:, self.contfeats],
                                x1[:, self.binfeats], x2[:, self.binfeats], x3[:, self.binfeats]], axis=1)
            contfeats, binfeats = range(0, 3*len(self.contfeats)), range(3*len(self.contfeats), 3 * 25)
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, contfeats, binfeats


class IHDPconf(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=1000, extra_feats=1):
        self.path_data = path_data
        self.replications = replications
        self.extra_feats = extra_feats

    def generate_y(self, x, t, seed=1):
        np.random.seed(seed)
        w = np.random.uniform(size=(x.shape[1], self.extra_feats)) + .1 * np.random.randn(x.shape[1], self.extra_feats)
        xnew = np.concatenate([x, np.cos(np.dot(x, w))], axis=1)
        # run k-means on [x, t] to get z and use y

        beta_B = np.random.choice(np.arange(x.shape[1]), size=5, replace=False)
        mask = np.zeros((1, xnew.shape[1]))
        # switch on features that got selected by the mask
        mask[0, beta_B] = 1
        # switch on features that are confounding
        mask[0, -self.extra_feats:] = 1
        # this does not select only five features; it seems to me that it selects integers from 1 to 5,
        # 25 times with replacement according to the given probablities. Ask Uri about this
        # beta_B = (randsample(5, d, true, [0.6; 0.1; 0.1; 0.1; 0.1])-1) / 10;
        # mu0 = exp((X + 0.5) * beta_B);
        # mu1 = X * beta_B;
        # omega = mean(mu1(T == 0) - mu0(T == 0)) - 4;
        # mu1 = mu1 - omega;
        # yB0 = mu0 + randn(n, 1);
        # yB1 = mu1 + randn(n, 1);
        # Y = nan(size(T));
        # Y(T == 1) = yB1(T == 1);
        # Y(T == 0) = yB0(T == 0);

        weights = np.concatenate([np.random.uniform(size=(x.shape[1], 1)),
                                  np.random.uniform(low=1., high=2., size=(self.extra_feats, 1))], axis=0) / 5.
        mu0 = np.exp(np.dot(xnew * mask + .5, weights))
        mu1 = np.dot(xnew * mask, weights)
        omega = np.mean(mu1[t == 0] - mu0[t == 0]) - 4
        mu1 = mu1 - omega
        yB0 = mu0 + np.random.randn(x.shape[0], 1)
        yB1 = mu1 + np.random.randn(x.shape[0], 1)
        y, y_cf = np.zeros_like(t), np.zeros_like(t)
        y[t == 1] = yB1[t == 1]
        y[t == 0] = yB0[t == 0]
        y_cf[t == 1] = yB0[t == 1]
        y_cf[t == 0] = yB1[t == 0]

        return y, y_cf, mu0, mu1

    def __iter__(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            # t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            # mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            x, t = data[:, 5:], data[:, 0]
            y, y_cf, mu_0, mu_1 = self.generate_y(x, t, seed=i+1)
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            # t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            # mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            x, t = data[:, 5:], data[:, 0][:, np.newaxis]
            y, y_cf, mu_0, mu_1 = self.generate_y(x, t, seed=i + 1)
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test


class Jobs(object):
    def __init__(self, path_data="datasets/Jobs/"):
        self.path_data = path_data
        self.binfeats = [2, 3, 4, 5, 13, 14, 16]
        self.contfeats = [i for i in xrange(17) if i not in self.binfeats]

    # def get_data(self):
    #     data = np.load(self.path_data + 'jobs_DW_bin.npz')
    #     x, t, y, e = np.squeeze(data['x']), data['t'], data['yf'].astype(np.float32), data['e'].ravel()
    #     return (x, t, y), e
    #     # data = np.loadtxt(self.path_data + 'jobs_DW_bin.csv', delimiter=',')
    #     # t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
    #     # mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    #     # return (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_test(self):
        train = np.load(self.path_data + 'jobs_DW_bin.train.npz')
        test = np.load(self.path_data + 'jobs_DW_bin.test.npz')
        xtrain, ttrain, ytrain, etrain = np.squeeze(train['x']), train['t'], train['yf'].astype(np.float32), train['e'].ravel()
        xtest, ttest, ytest, etest = np.squeeze(test['x']), test['t'], test['yf'].astype(np.float32), test['e'].ravel()
        return (xtrain, ttrain, ytrain, etrain), (xtest, ttest, ytest, etest)

    def get_train_valid_test(self, replications=50):
        train = np.load(self.path_data + 'jobs_DW_bin.train.npz')
        test = np.load(self.path_data + 'jobs_DW_bin.test.npz')
        xtrain, ttrain, ytrain, etrain = np.squeeze(train['x']), train['t'], train['yf'].astype(np.float32), train['e'].ravel()
        xtest, ttest, ytest, etest = np.squeeze(test['x']), test['t'], test['yf'].astype(np.float32), test['e'].ravel()
        valid_idx = np.load(self.path_data + 'validation_sets.npz')['validation_sets']
        for i in range(valid_idx.shape[0])[0:replications]:
            v_idx = valid_idx[i]
            t_idx = np.array([j for j in xrange(xtrain.shape[0]) if j not in v_idx])
            yield (xtrain[t_idx], ttrain[t_idx], ytrain[t_idx], etrain[t_idx]), \
                  (xtrain[v_idx], ttrain[v_idx], ytrain[v_idx], etrain[v_idx]), \
                  (xtest, ttest, ytest, etest), self.contfeats, self.binfeats


class Jobs3(object):
    def __init__(self, path_data="datasets/Jobs/", noise_cont=0.2, noise_bin=0.2):
        self.path_data = path_data
        self.noise_cont = noise_cont
        self.noise_bin = noise_bin
        self.binfeats = [2, 3, 4, 5, 13, 14, 16]
        self.contfeats = [i for i in xrange(17) if i not in self.binfeats]

    def get_train_test(self):
        train = np.load(self.path_data + 'jobs_DW_bin.train.npz')
        test = np.load(self.path_data + 'jobs_DW_bin.test.npz')
        xtrain, ttrain, ytrain, etrain = np.squeeze(train['x']), train['t'], train['yf'].astype(np.float32), train['e'].ravel()
        xtest, ttest, ytest, etest = np.squeeze(test['x']), test['t'], test['yf'].astype(np.float32), test['e'].ravel()

        # train views
        xtr1, xtr2, xtr3 = np.zeros_like(xtrain), np.zeros_like(xtrain), np.zeros_like(xtrain)
        # view1
        xtr1[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr1[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # view2
        xtr2[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr2[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # view3
        xtr3[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr3[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # noisy view of xtrain
        xtrain = np.concatenate([xtr1, xtr2, xtr3], axis=1)

        # test views
        xte1, xte2, xte3 = np.zeros_like(xtest), np.zeros_like(xtest), np.zeros_like(xtest)
        # view1
        xte1[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte1[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # view2
        xte2[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte2[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # view3
        xte3[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte3[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # noisy view of xtest
        xtest = np.concatenate([xte1, xte2, xte3], axis=1)

        return (xtrain, ttrain, ytrain, etrain), (xtest, ttest, ytest, etest)

    def get_train_valid_test(self, replications=50):
        train = np.load(self.path_data + 'jobs_DW_bin.train.npz')
        test = np.load(self.path_data + 'jobs_DW_bin.test.npz')
        xtrain, ttrain, ytrain, etrain = np.squeeze(train['x']), train['t'], train['yf'].astype(np.float32), train['e'].ravel()
        xtest, ttest, ytest, etest = np.squeeze(test['x']), test['t'], test['yf'].astype(np.float32), test['e'].ravel()

        # train views
        xtr1, xtr2, xtr3 = np.zeros_like(xtrain), np.zeros_like(xtrain), np.zeros_like(xtrain)
        # view1
        xtr1[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr1[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # view2
        xtr2[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr2[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # view3
        xtr3[:, self.contfeats] = xtrain[:, self.contfeats] + self.noise_cont * np.random.randn(xtrain.shape[0], len(self.contfeats))
        xtr3[:, self.binfeats] = xtrain[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtrain.shape[0], len(self.binfeats)))
        # noisy view of xtrain
        xtrain = np.concatenate([xtr1, xtr2, xtr3], axis=1)

        # test views
        xte1, xte2, xte3 = np.zeros_like(xtest), np.zeros_like(xtest), np.zeros_like(xtest)
        # view1
        xte1[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte1[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # view2
        xte2[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte2[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # view3
        xte3[:, self.contfeats] = xtest[:, self.contfeats] + self.noise_cont * np.random.randn(xtest.shape[0], len(self.contfeats))
        xte3[:, self.binfeats] = xtest[:, self.binfeats] * np.random.binomial(1, p=1 - self.noise_bin, size=(xtest.shape[0], len(self.binfeats)))
        # noisy view of xtest
        xtest = np.concatenate([xte1, xte2, xte3], axis=1)

        contfeats = self.contfeats + [i + 17 for i in self.contfeats] + [i + 2*17 for i in self.contfeats]
        binfeats = self.binfeats + [i + 17 for i in self.binfeats] + [i + 2*17 for i in self.binfeats]
        valid_idx = np.load(self.path_data + 'validation_sets.npz')['validation_sets']
        for i in range(valid_idx.shape[0])[0:replications]:
            v_idx = valid_idx[i]
            t_idx = np.array([j for j in xrange(xtrain.shape[0]) if j not in v_idx])
            yield (xtrain[t_idx], ttrain[t_idx], ytrain[t_idx], etrain[t_idx]), \
                  (xtrain[v_idx], ttrain[v_idx], ytrain[v_idx], etrain[v_idx]), \
                  (xtest, ttest, ytest, etest), contfeats, binfeats



class ACIC(object):
    def __init__(self, path_data="datasets/ACIC", replications=100, n_outcomes=20):
        self.path_data = path_data
        self.replications = replications
        self.n_outcomes = n_outcomes

    def __iter__(self):
        from string import ascii_uppercase
        dic = {c: float(i + 1) for i, c in enumerate(ascii_uppercase)}
        conv = lambda s: dic[s[1]]
        x = np.genfromtxt(self.path_data + '/x.csv', skip_header=1, delimiter=',', dtype=float,
                          converters={1: conv, 20: conv, 23: conv})
        for i in xrange(self.n_outcomes):
            zy = np.loadtxt(self.path_data + '/zy_' + str(i+1) + '.csv', skiprows=1, delimiter=',')
            t, y = zy[:, 0], zy[:, 1][:, np.newaxis]
            yield (x, t, y)


class News(object):
    def __init__(self, path_data="datasets/News/csv", replications=50):
        self.path_data = path_data
        self.replications = replications

    def __iter__(self):
        for i in xrange(self.replications):
            fname = self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv'
            sparse_x = np.loadtxt(fname + '.x', delimiter=',', dtype=int)
            x = np.zeros((sparse_x[0, 0], sparse_x[0, 1]))
            x[sparse_x[1:, 0] - 1, sparse_x[1:, 1] - 1] = sparse_x[1:, 2]
            ty = np.loadtxt(fname + '.y', delimiter=',')
            t, y, y_cf, mu_0, mu_1 = ty[:, 0], ty[:, 1][:, np.newaxis], ty[:, 2][:, np.newaxis], ty[:, 3], ty[:, 4]
            yield (x, t, y), (y_cf, mu_0, mu_1)


class MNISTconf(object):
    def __init__(self, type_data='perm_inv', seed=1):
        self.type_data = type_data
        self.seed = seed

    def load_mnist(self):
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        xtrain = xtrain / 255.
        xtest = xtest / 255.
        if self.type_data == 'perm_inv':
            xtrain, xtest = xtrain.reshape(xtrain.shape[0], -1), xtest.reshape(xtest.shape[0], -1)
        ytrain_k, ytest_k = to_categorical(ytrain, 10), to_categorical(ytest, 10)
        return (xtrain, ytrain_k), (xtest, ytest_k)

    def get_ty(self):
        (xtrain, ytrain), (xtest, ytest) = self.load_mnist()
        np.random.seed(self.seed)
        wt = np.random.uniform(size=(10, 1))
        pt_train = 1. / (1. + np.exp(- np.dot(ytrain, wt)))
        pt_test = 1. / (1. + np.exp(- np.dot(ytest, wt)))
        t_train, t_test = np.random.binomial(1, pt_train), np.random.binomial(1, pt_test)

        wy_t0, wy_t1 = np.random.uniform(size=(11, 1)), np.random.uniform(size=(11, 1))
        py_t0_train = 1. / (1. + np.exp(- np.dot(np.concatenate([ytrain, t_train], axis=1), wy_t0)))
        py_t1_train = 1. / (1. + np.exp(- np.dot(np.concatenate([ytrain, t_train], axis=1), wy_t1)))
        py_t0_test = 1. / (1. + np.exp(- np.dot(np.concatenate([ytest, t_test], axis=1), wy_t0)))
        py_t1_test = 1. / (1. + np.exp(- np.dot(np.concatenate([ytest, t_test], axis=1), wy_t1)))

        y_1_train, y_0_train = np.random.binomial(1, py_t1_train), np.random.binomial(1, py_t0_train)
        y_1_test, y_0_test = np.random.binomial(1, py_t1_test), np.random.binomial(1, py_t0_test)
        y_o_train = t_train * y_1_train + (1 - t_train) * y_0_train
        y_o_test = t_test * y_1_test + (1 - t_test) * y_0_test

        return (xtrain, t_train, y_o_train, np.argmax(ytrain, 1)), (xtest, t_test, y_o_test, np.argmax(ytest, 1)), \
               ((y_1_train, py_t1_train), (y_0_train, py_t0_train)), ((y_1_test, py_t1_test), (y_0_test, py_t0_test))

    def get_train_test(self):
        return self.get_ty()
