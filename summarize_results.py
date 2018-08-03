import numpy as np
from scipy.stats import sem
import os

def load_run_pehe(path, pehe_type='test'):
    pehe = np.loadtxt(path, delimiter=',')
    if (pehe_type== 'train'):
        pehe=pehe[:,3]
    elif (pehe_type == 'test'):
        pehe=pehe[:,6]
    avg_pehe=np.average(pehe)
    sem_pehe=sem(pehe)
    return avg_pehe, sem_pehe

def split(s, sep):
    parts = []
    bracket_level = 0
    current = []
    for c in (s + sep):
        if c == sep and bracket_level == 0:
            parts.append("".join(current))
            current = []
        else:
            if c in "([{":
                bracket_level += 1
            elif c in "}])":
                bracket_level -= 1
            current.append(c)
    return parts

def load_run_configs(path_used_configs,run_num):
    config_file=open(path_used_configs,'r')
    str_configs=config_file.readlines()[run_num-1].rstrip()
    config_file.close()
    return dict([tuple(pair.split(':')) for pair in split(str_configs,',')])

def parse_config(config_str):
    try:
        return int(config_str)
    except ValueError:
        try:
            return float(config_str)
        except ValueError:
            return config_str


def load_pehe_sorted_by_config(path_result_unformatted,path_used_configs, pehe_type='test'):
    configs_dict=load_run_configs(path_used_configs,0) #bad form
    for key in configs_dict.keys():
        configs_dict[key]={}

    i_run = 1
    while os.path.isfile(path_result_unformatted % i_run):
        avg_pehe,sem_pehe=load_run_pehe(path_result_unformatted % i_run, pehe_type)
        run_configs=load_run_configs(path_used_configs,i_run)
        for key in configs_dict.keys():
            config_value=parse_config(run_configs[key])
            if config_value not in configs_dict[key].keys():
                configs_dict[key][config_value]=[]
            configs_dict[key][config_value].append(avg_pehe)
        i_run+=1

    return configs_dict

def save_pehe_sorted_by_config(path_to_save_unformatted,path_result_unformatted,path_used_configs, pehe_type='test'):
    configs_dict=load_pehe_sorted_by_config(path_result_unformatted,path_used_configs, pehe_type)
    for config_name in configs_dict.keys():
        if len(configs_dict[config_name]) > 1:
            str_csv='value,pehe_median,pehe_avg,pehe_stdev,pehe_individual\n'
            for config_value in sorted(configs_dict[config_name].keys()):
                str_csv += str(config_value)+','+ \
                           '%.3f' % (np.median(configs_dict[config_name][config_value]))+','+ \
                           '%.3f' % (np.average(configs_dict[config_name][config_value]))+','+ \
                           '%.3f' % (np.std(configs_dict[config_name][config_value]))+','
                for pehe in sorted(configs_dict[config_name][config_value]):
                    str_csv += '%.3f' % pehe + ','
                str_csv += '\n'
            path_to_save = path_to_save_unformatted % config_name
            result_file = open(path_to_save,'w')
            result_file.write(str_csv)
            result_file.close()

def save_config_results_summary(path_to_save,path_result_unformatted,path_used_configs, pehe_type='test', sort_by='avg'):
    configs_dict=load_pehe_sorted_by_config(path_result_unformatted,path_used_configs, pehe_type)
    str_csv = 'config_name,1st_value,pehe_%s,pehe_stdev,2nd_value,pehe_%s,pehe_stdev\n'% (sort_by,sort_by)
    for config_name in configs_dict.keys():
        if len(configs_dict[config_name]) > 1:
            for config_value in sorted(configs_dict[config_name].keys()):
                if sort_by in ('avg', 'average'):
                    configs_dict[config_name][config_value] = (np.average(configs_dict[config_name][config_value]),
                                                               np.std(configs_dict[config_name][config_value]))
                elif sort_by in ('med', 'median'):
                    configs_dict[config_name][config_value] = (np.median(configs_dict[config_name][config_value]),
                                                               np.std(configs_dict[config_name][config_value]))
                elif sort_by in ('min', 'minimum', 'best'):
                    configs_dict[config_name][config_value] = (np.min(configs_dict[config_name][config_value]), 0)

            str_csv += config_name + ','
            for config_value, (pehe_avg, pehe_stdev) in sorted(configs_dict[config_name].items(), key=lambda tup: tup[1][0]):
                str_csv += str(config_value) + ',' + '%.3f' % pehe_avg + ',' + '%.3f' % pehe_stdev +','
            str_csv += '\n'
    result_file = open(path_to_save,'w')
    result_file.write(str_csv)
    result_file.close()

results_folder='results/add_ipm_regularizer/with_ipm_hypar_search_2/'
save_pehe_sorted_by_config(results_folder+'/effect_of_%s_on_pehe_test.csv',results_folder+'/run%d.csv',results_folder+'/used_configs.txt', 'test')
save_config_results_summary(results_folder+'/summary_of_configs_effect_on_pehe_test_min.csv',results_folder+'/run%d.csv',results_folder+'/used_configs.txt', pehe_type='test', sort_by='min')

