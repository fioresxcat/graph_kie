import os
import re
import pdb
import json
import math
import numpy as np
import unidecode
from bpemb import BPEmb
from pathlib import Path
from typing import List, Tuple
import copy
from model import *
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_output_dir = 'model_output'
    res = {}
    ls_exp_name = []
    for report_fp in Path(model_output_dir).rglob('report.txt'):
        martname = report_fp.parent.parent.parent.name
        if martname not in res:
            res[martname] = {}
        split = report_fp.parent.name
        if split not in res[martname]:
            res[martname][split] = {}
        exp_name = report_fp.parent.parent.name
        exp_name = '_'.join(exp_name.split('_')[1:])
        ls_exp_name.append(exp_name)
        if exp_name not in res[martname][split]:
            res[martname][split][exp_name] = {}
        with open(report_fp) as f:
            report = f.readlines()
        
        f1_line = [line for line in report if 'overall f1 score:' in line][0]
        f1 = float(f1_line.split(':')[-1].strip())

        precision_line = [line for line in report if 'overall precision:' in line][0]
        precision = float(precision_line.split(':')[-1].strip())

        recall_line = [line for line in report if 'overall recall:' in line][0]
        recall = float(recall_line.split(':')[-1].strip())

        err_line = [line for line in report if 'num_err:' in line][0]
        err = int(err_line.split(':')[-1].strip())

        total_line = [line for line in report if 'num_total:' in line][0]
        total = int(total_line.split(':')[-1].strip())

        res[martname][split][exp_name]['f1'] = f1
        res[martname][split][exp_name]['precision'] = precision
        res[martname][split][exp_name]['recall'] = recall
        res[martname][split][exp_name]['err'] = err
        res[martname][split][exp_name]['total'] = total
    
    # pdb.set_trace()
    res_final = {}
    for martname in res:
        for split in res[martname]:
            for exp_name in sorted(res[martname][split].keys()):
                if f'{martname}_{split}' not in res_final:
                    res_final[f'{martname}_{split}'] = {}
                    res_final[f'{martname}_{split}']['f1'] = [res[martname][split][exp_name]['f1']]
                    res_final[f'{martname}_{split}']['precision'] = [res[martname][split][exp_name]['precision']]
                    res_final[f'{martname}_{split}']['recall'] = [res[martname][split][exp_name]['recall']]
                    res_final[f'{martname}_{split}']['err'] = [res[martname][split][exp_name]['err']]
                    res_final[f'{martname}_{split}']['total'] = [res[martname][split][exp_name]['total']]
                else:
                    res_final[f'{martname}_{split}']['f1'].append(res[martname][split][exp_name]['f1'])
                    res_final[f'{martname}_{split}']['precision'].append(res[martname][split][exp_name]['precision'])
                    res_final[f'{martname}_{split}']['recall'].append(res[martname][split][exp_name]['recall'])
                    res_final[f'{martname}_{split}']['err'].append(res[martname][split][exp_name]['err'])
                    res_final[f'{martname}_{split}']['total'].append(res[martname][split][exp_name]['total'])
    

    # pdb.set_trace()
                    
    # # plot metric for all martname_split in res_final in a single plot, each martname_split is a group, each group has multiple bar for each exp_name
    # for metric in ['f1', 'precision', 'recall']:
    #     plt.figure()
    #     keys = list(res_final.keys())
    #     values = np.array(list([res_final[key][metric] for key in keys]))
    #     n_groups, n_bars = values.shape
    #     bar_width = 0.2
    #     fig, ax = plt.subplots()
    #     xticks = np.arange(n_groups) * (n_bars + 1) + bar_width / 2
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(keys)
    #     for i in range(n_bars):
    #         x = np.arange(n_groups) * (n_bars + 1) + i * bar_width
    #         ax.bar(x, values[:, i], bar_width, label=f'{list(res_final.keys())[i]}')
    #     ax.legend()
    #     plt.title(metric)
    #     plt.savefig(f'{metric}.png')
    #     plt.close()

    # # write f1 result to csv file
    # ls_exp_name = sorted(list(set(ls_exp_name)))
    # metric = 'f1'
    # with open(f'model_output/{metric}.csv', 'w') as f:
    #     f.write(f',{",".join(ls_exp_name)}\n')
    #     for martname in res_final.keys():
    #         scores = res_final[martname][metric]
    #         f.write(f'{martname},{",".join([str(score) for score in scores])}\n')
    

    # write err/total result to csv file
    ls_exp_name = sorted(list(set(ls_exp_name)))
    metric = 'err'
    with open(f'error.csv', 'w') as f:
        f.write(f',{",".join(ls_exp_name)}\n')
        for martname in res_final.keys():
            scores = res_final[martname][metric]
            f.write(f'{martname},{",".join([str(score) for score in scores])}\n')
