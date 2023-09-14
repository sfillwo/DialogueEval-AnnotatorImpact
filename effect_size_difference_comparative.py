import math

import pandas as pd
import random
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import bootstrap
from numpy import mean
from numpy import var
from math import sqrt
import numpy as np
import itertools

from utilities.analysis import *
from utilities.mappings import bot_transformer
from utilities.graphing import grouped_barplot_benchmark, grouped_barplot_cohensd
from effect_size import effect_size_comparative

PLOT = False

random.seed(654)

#########################################
##  DATA
#########################################

# Ui
ui = data.dialogue_collection.annotation_dataframe()
ui_comparative = data.dialogue_collection.comparative_annotation_dataframe()

# Sx
sx = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123)
sx_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

# Developers
developer = get_singly_annotated(developer_ext.student_external.annotation_dataframe(), seed=123)
developer_comparative = get_singly_annotated(developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

# Non developers
nondeveloper = get_singly_annotated(non_developer_ext.student_external.annotation_dataframe(), seed=123)
nondeveloper_comparative = get_singly_annotated(non_developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

all_bots = sx.index.unique('bot')
all_bot_pairs = list(itertools.combinations(all_bots, 2))
labels = sx.xs('likert dialogue', level='category').index.unique('label').tolist()


def preprocess_comparative(df_dict):
    for name, evaluation in df_dict.items():
        evaluation.columns = ['score']
        df_dict[name] = evaluation.reset_index().pivot(index=('bot', 'bot comp', 'dialogues'), columns='label', values='score').sort_index()
    return df_dict

def switch_val(item):
    old_vals = [-1, 1]
    new_vals = [1, -1]
    return new_vals[old_vals.index(item)] if item in old_vals else item
val_switcher = np.vectorize(switch_val)

def avg_effect_size_comparative(
        group1_botpair1, group1_botpair2, group1_botpair3, group1_botpair4,  group1_botpair5, group1_botpair6,
        group2_botpair1, group2_botpair2, group2_botpair3, group2_botpair4, group2_botpair5, group2_botpair6,
        axis=None):
    group1_botpairs = [group1_botpair1, group1_botpair2, group1_botpair3, group1_botpair4,  group1_botpair5, group1_botpair6]
    group2_botpairs = [group2_botpair1, group2_botpair2, group2_botpair3, group2_botpair4, group2_botpair5, group2_botpair6]
    group1, group2 = {}, {}
    for i, (bot1, bot2) in enumerate(all_bot_pairs):
        group1_results = group1_botpairs[i]
        group1.setdefault(bot1, []).append(group1_results)
        group1.setdefault(bot2, []).append(val_switcher(group1_results))

        group2_results = group2_botpairs[i]
        group2.setdefault(bot1, []).append(group2_results)
        group2.setdefault(bot2, []).append(val_switcher(group2_results))

    concat_group1, concat_group2 = {}, {}
    for bot, samples in group1.items():
        concat_axis = None if len(np.array(samples[0]).shape) == 1 else 1
        concat_group1[bot] = np.concatenate(samples, axis=concat_axis)
        concat_group2[bot] = np.concatenate(group2[bot], axis=concat_axis)

    group1_bots = [concat_group1[bot] for bot in all_bots]
    group2_bots = [concat_group2[bot] for bot in all_bots]
    bot_pairs = list(combinations(range(4), 2))
    botpair_strs = [(all_bots[idx1], all_bots[idx2]) for idx1, idx2 in bot_pairs]
    diffs = []
    for idx1, idx2 in bot_pairs:
        diff = effect_size_comparative(group1_bots[idx1], group1_bots[idx2], group2_bots[idx1], group2_bots[idx2])
        diffs.append(diff)
    diffs_np = np.array(diffs)
    means = np.mean(diffs_np, axis=0)
    sorted_means = np.sort(means).tolist()
    if len(sorted_means) == 1:
        return sorted_means[0]
    return sorted_means

def run_bootstrap_on(df_dict, n_resamples, confidence_level, prep_func, stat_func, bots=None):
    if bots is None:
        bots = all_bots
    df_dict = prep_func(df_dict)
    group_pairs = [('ui', 'sx'), ('dev', 'ui'), ('dev', 'sx'), ('dev', 'ndev'),('ui', 'ndev'), ('ndev', 'sx')]
    results = {}
    for label in labels:
        for group1, group2 in group_pairs:
            group1_ratings = [df_dict[group1].xs(bot_pair[0], level='bot').xs(bot_pair[1], level='bot comp')[label].tolist() for bot_pair in all_bot_pairs]
            group2_ratings = [df_dict[group2].xs(bot_pair[0], level='bot').xs(bot_pair[1], level='bot comp')[label].tolist() for bot_pair in all_bot_pairs]
            result = bootstrap([*group1_ratings, *group2_ratings], stat_func,
                               n_resamples=n_resamples, confidence_level=confidence_level,
                               method='percentile', random_state=654)
            point_estimate = stat_func(*group1_ratings, *group2_ratings)
            # print(label, group1, group2, point_estimate)
            results.setdefault(label, {})[(group1, group2)] = (result, point_estimate)
    return results


#####################################################################

rounds = 10000
k = 100

#####################################################################

# Novice results

print('Pairwise - All bots')
start = time.time()
result = run_bootstrap_on({'ui': ui_comparative, 'sx': sx_comparative, 'dev': developer_comparative, 'ndev': nondeveloper_comparative},
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_effect_size_comparative)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")

group_mapping = {
    'ui': 'Stu$_i$',
    'sx': 'Sur$_x$',
    'dev': 'Dev$_x$',
    'ndev': 'Stu$_x$',
}

results_entries = []
for label, label_results in result.items():
    for groups, r in label_results.items():
        if True: #'dev' not in groups:
            updated_groups = []
            for g in groups:
                updated_groups += [group_mapping[g]]
            updated_groups = '/'.join(sorted(updated_groups))
            results_entries.append([label, updated_groups, r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
results_df = pd.DataFrame.from_records(results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
grouped_barplot_cohensd(results_df, None, "", "", ylim=(0, 1.0), value_col="estimate", rot=0,
                        filename='outputs/figures/effect_size_difference_comparative_percnetile', plot_err=True,
                        fig_size=(20,3), width=0.8)


