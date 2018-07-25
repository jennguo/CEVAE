import numpy as np
from scipy.stats import sem
import os

def load_run_pehe(path, type='test'):
    pehe = np.loadtxt(path, delimiter=',')
    if (type=='train'):
        pehe=pehe[:,3]
    elif (type=='test'):
        pehe=pehe[:,6]
    avg_pehe=np.average(pehe)
    sem_pehe=sem(pehe)
    return avg_pehe, sem_pehe

def load_run_configs(path_used_configs,run_num):
    config_file=open(path_used_configs,'r')
    str_configs=config_file.readlines()[run_num-1].rstrip()
    config_file.close()
    return dict([tuple(pair.split(':')) for pair in str_configs.split(',')])

def parse_config(config_str):
    try:
        return int(config_str)
    except ValueError:
        try:
            return float(config_str)
        except ValueError:
            return config_str


def load_pehe_sorted_by_config(path_result_unformatted,path_used_configs):
    configs_dict=load_run_configs(path_used_configs,0) #bad form
    for key in configs_dict.keys():
        configs_dict[key]={}

    i_run = 1
    while os.path.isfile(path_result_unformatted % i_run):
        avg_pehe,sem_pehe=load_run_pehe(path_result_unformatted % i_run)
        run_configs=load_run_configs(path_used_configs,i_run)
        for key in configs_dict.keys():
            config_value=parse_config(run_configs[key])
            if config_value not in configs_dict[key].keys():
                configs_dict[key][config_value]=[]
            configs_dict[key][config_value].append(avg_pehe)
        i_run+=1

    return configs_dict

def save_pehe_sorted_by_config(path_to_save_unformatted,path_result_unformatted,path_used_configs):
    configs_dict=load_pehe_sorted_by_config(path_result_unformatted,path_used_configs)
    for key in configs_dict.keys():
        str_csv='value,pehe_median,pehe_avg,pehe_stdev,pehe_individual\n'
        for config_value in sorted(configs_dict[key].keys()):
            str_csv += str(config_value)+','+ \
                       '%.3f' % (np.median(configs_dict[key][config_value]))+','+ \
                       '%.3f' % (np.average(configs_dict[key][config_value]))+','+ \
                       '%.3f' % (np.std(configs_dict[key][config_value]))+','
            for pehe in sorted(configs_dict[key][config_value]):
                str_csv += '%.3f' % pehe + ','
            str_csv += '\n'
        path_to_save = path_to_save_unformatted % key
        result_file = open(path_to_save,'w')
        result_file.write(str_csv)
        result_file.close()

save_pehe_sorted_by_config('results (copy)/effect_of_%s_on_pehe_test.csv','results (copy)/run%d.csv','results (copy)/used_configs.txt')