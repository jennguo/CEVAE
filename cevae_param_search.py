#todo sort imports
from __future__ import absolute_import

import numpy as np
from argparse import ArgumentParser
from datasets import IHDP

import cevae_ihdp

import os

import gc

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    return cfg_sample

def cfg_string(cfg):
    ks = sorted(cfg.keys())
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()

def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs

# helper
def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs

# helper
def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)

def run(cfg_file, num_runs):
    configs = load_config(cfg_file)

    outdir = configs['outdir'][0]
    used_cfg_file = '%s/used_configs.txt' % outdir

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    ##hardcoded
    first_unused_run_number=0
    while os.path.isfile(outdir + '/run%d.csv' % (first_unused_run_number+1)):
        first_unused_run_number+=1

    for i in range(first_unused_run_number,first_unused_run_number+num_runs):
        cfg = sample_config(configs)
        while is_used_cfg(cfg, used_cfg_file):
            print 'Configuration used, skipping: %s' % str(cfg)
            cfg = sample_config(configs)

        print '------------------------------'
        print 'Run %d of %d:' % (i+1, num_runs)
        print '------------------------------'
        print '\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.iteritems() if len(configs[k])>1])

        parser = ArgumentParser()

        parser.add_argument('-reps', type=int, default=cfg['reps'])  # was 10 todo change name 'reps' to 'n_experiments'
        parser.add_argument('-earl', type=int, default=cfg['earl'])  # was 10
        parser.add_argument('-lr', type=float, default=cfg['learning_rate'])  # was 0.001
        parser.add_argument('-opt', choices=['adam', 'adamax'], default=cfg['optimizer'])  # was 'adam'; has no effect!
        parser.add_argument('-epochs', type=int, default=cfg['epochs'])  # was 100
        parser.add_argument('-print_every', type=int, default=cfg['print_every'])

        args = parser.parse_args()
        args.true_post = True

        path_data_unformatted = cfg['path_data_unformatted']
        bin_feats = cfg['bin_feats']
        dim_x=cfg['dim_x']

        dataset = IHDP(path_data_unformatted=path_data_unformatted, dim_x=dim_x, replications=args.reps, bin_feats=bin_feats) # todo add cfg options for path_data, binfeats, contfeats, etc.

        scores = np.zeros((args.reps, 3))
        scores_test = np.zeros((args.reps, 3))

        M = None  # batch size during training
        d = cfg['latent_dim']  # latent dimension; was 20
        lamba = cfg['lambda'] # weight decay; was 1e-4
        nh, h = cfg['n_hidden'], cfg['size_hidden']  # number and size of hidden layers; was 3 and 200

        cevae_ihdp.run(args,dataset,scores,scores_test,M,d,lamba,nh,h,i+1,outdir)

        save_used_cfg(cfg, used_cfg_file)

if __name__ == "__main__":
    #gc.set_debug(gc.DEBUG_LEAK)
    run("config.txt", 1)
    # "config.txt" = where the configs are
    # 1 = number of runs. It's 1 because if you run multiple runs, for some reason the memory keeps increasing
    # (memory leaking? global variable increasing in size?) to the point where the process is killed.
    # So the current workaround is to run one run inside this script, but to call this script
    # many times if you want many runs.
    # todo: incorporate the parameters into the sys.args
