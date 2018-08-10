#todo sort imports
from __future__ import absolute_import

import numpy as np
from argparse import ArgumentParser
from datasets import IHDP

import cevae_ihdp

import os

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

    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

        parser.add_argument('-outdir', type=str, default=outdir)
        parser.add_argument('-reps_begin', type=int, default=cfg['reps_begin'])
        parser.add_argument('-reps_end', type=int, default=cfg['reps_end'])  # was 10 todo change name 'reps' to 'n_experiments'
        parser.add_argument('-epochs', type=int, default=cfg['epochs'])  # was 100
        parser.add_argument('-print_every', type=int, default=cfg['print_every'])
        parser.add_argument('-earl', type=int, default=cfg['earl'])  # was 10
        parser.add_argument('-lr', type=float, default=cfg['learning_rate'])  # was 0.001
        parser.add_argument('-latent_dim', type=int, default=cfg['latent_dim'])
        parser.add_argument('-lamba', type=float, default=cfg['lambda']) # weight decay; was 1e-4
        parser.add_argument('-n_hidden', type=int, default=cfg['n_hidden'])  # number of hidden layers; was 3
        parser.add_argument('-size_hidden', type=int, default=cfg['size_hidden']) # size of hidden layers; was 200

        parser.add_argument('-use_cfrnet_structure', type=bool, default=cfg['use_cfrnet_structure'])
        parser.add_argument('-cfr_n_phi', type=int, default=cfg['cfr_n_phi'])
        parser.add_argument('-cfr_n_mu', type=int, default=cfg['cfr_n_mu'])
        parser.add_argument('-wass_alpha', type=float, default=cfg['wass_alpha'])
        parser.add_argument('-wass_lambda', type=float, default=cfg['wass_lambda'])
        parser.add_argument('-wass_iterations', type=float, default=cfg['wass_iterations'])
        parser.add_argument('-wass_bpt', type=bool, default=cfg['wass_bpt'])
        parser.add_argument('-wass_use_p_correction', type=bool, default=cfg['wass_use_p_correction'])

        args = parser.parse_args()
        args.true_post = True

        path_data_unformatted = cfg['path_data_unformatted']
        bin_feats = cfg['bin_feats']
        dim_x = cfg['dim_x']
        dataset = IHDP(path_data_unformatted=path_data_unformatted, dim_x=dim_x, reps_begin=args.reps_begin, reps_end=args.reps_end, bin_feats=bin_feats) # todo add cfg options for path_data, binfeats, contfeats, etc.

        scores = np.zeros((args.reps_end-args.reps_begin+1, 3))
        scores_test = np.zeros((args.reps_end-args.reps_begin+1, 3))

        M = None  # batch size during training

        cevae_ihdp.run(args,dataset,scores,scores_test,M,i+1)

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
