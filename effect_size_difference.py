import pandas as pd
import random
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import bootstrap
import numpy as np
from utilities.analysis import *
from utilities.mappings import bot_transformer
from utilities.graphing import grouped_barplot_benchmark, grouped_barplot_cohensd, grouped_barplot_effectsize
from effect_size import effect_size_likert, effect_size_comparative

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

# Ux
ux = get_singly_annotated(data_student_ext.student_external.annotation_dataframe(), seed=123)
ux_comparative = get_singly_annotated(data_student_ext.student_external.comparative_annotation_dataframe(), seed=123)

all_bots = sx.index.unique('bot')
labels = sx.xs('likert dialogue', level='category').index.unique('label').tolist()

def preprocess_likert(df_dict):
    for name, evaluation in df_dict.items():
        evaluation.columns = ['score']
        df_dict[name] = evaluation.xs('likert dialogue', level='category').reset_index().pivot(index=('bot', 'item'), columns='label', values='score').sort_index()
    return df_dict

def preprocess_comparative(df_dict):
    for name, evaluation in df_dict.items():
        evaluation.columns = ['score']
        df_dict[name] = evaluation.xs('comparative', level='category').reset_index().pivot(index=('bot', 'item'), columns='label', values='score').sort_index()
    return df_dict

def avg_effect_size_likert(group1_bot1, group1_bot2, group1_bot3, group1_bot4,
                       group2_bot1, group2_bot2, group2_bot3, group2_bot4,
                       axis=None):
    group1_bots = [group1_bot1, group1_bot2, group1_bot3, group1_bot4]
    group2_bots = [group2_bot1, group2_bot2, group2_bot3, group2_bot4]
    bot_pairs = list(combinations(range(4), 2))
    botpair_strs = [(all_bots[idx1], all_bots[idx2]) for idx1, idx2 in bot_pairs]
    diffs = []
    for idx1, idx2 in bot_pairs:
        diff = effect_size_likert(group1_bots[idx1], group1_bots[idx2], group2_bots[idx1], group2_bots[idx2])
        diffs.append(diff)
    diffs_np = np.array(diffs)
    means = np.mean(diffs_np, axis=0)
    return np.sort(means)

def avg_effect_size_comparative(group1_bot1, group1_bot2, group1_bot3, group1_bot4,
                       group2_bot1, group2_bot2, group2_bot3, group2_bot4,
                       axis=None):
    group1_bots = [group1_bot1, group1_bot2, group1_bot3, group1_bot4]
    group2_bots = [group2_bot1, group2_bot2, group2_bot3, group2_bot4]
    bot_pairs = list(combinations(range(4), 2))
    botpair_strs = [(all_bots[idx1], all_bots[idx2]) for idx1, idx2 in bot_pairs]
    diffs = []
    for idx1, idx2 in bot_pairs:
        diff = effect_size_comparative(group1_bots[idx1], group1_bots[idx2], group2_bots[idx1], group2_bots[idx2])
        diffs.append(diff)
    diffs_np = np.array(diffs)
    means = np.mean(diffs_np, axis=0)
    return np.sort(means)

def run_bootstrap_on(df_dict, n_resamples, confidence_level, prep_func, stat_func, bots=None):
    if bots is None:
        bots = all_bots
    df_dict = prep_func(df_dict)
    group_pairs = [('ui', 'sx')] #, ('ux', 'ui'), ('ux', 'sx')]
    results = {}
    for label in labels:
        for group1, group2 in group_pairs:
            group1_ratings = [df_dict[group1].xs(bot, level='bot')[label].tolist() for bot in bots]
            group2_ratings = [df_dict[group2].xs(bot, level='bot')[label].tolist() for bot in bots]
            result = bootstrap([*group1_ratings, *group2_ratings], stat_func,
                               n_resamples=n_resamples, confidence_level=confidence_level,
                               method='percentile', random_state=654)
            point_estimate = stat_func(*group1_ratings, *group2_ratings)
            results.setdefault(label, {})[(group1, group2)] = (result, point_estimate[0])
    return results


#####################################################################

rounds = 10000

#####################################################################

print('Likert - All bots')
start = time.time()
result = run_bootstrap_on({'ui': ui, 'sx': sx},
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_likert,
                          stat_func=avg_effect_size_likert)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")


likert_results_entries = []
for label, label_results in result.items():
    for groups, r in label_results.items():
        likert_results_entries.append(["likert", label, groups, r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
likert_results_df = pd.DataFrame.from_records(likert_results_entries, columns=["method", "label", "groups", "estimate", "CI low", "CI high"])
grouped_barplot_cohensd(likert_results_df, "d", "$\Delta$ Effect size", "Label", value_col="estimate", rot=0,
                        filename='outputs/figures/effect_size_difference_likert')


print('Comparative - All bots')
start = time.time()
result = run_bootstrap_on({'ui': ui, 'sx': sx},
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_effect_size_comparative)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")


comp_results_entries = []
for label, label_results in result.items():
    for groups, r in label_results.items():
        comp_results_entries.append(["pairwise", label, groups, r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
comp_results_df = pd.DataFrame.from_records(comp_results_entries, columns=["method", "label", "groups", "estimate", "CI low", "CI high"])
grouped_barplot_cohensd(comp_results_df, "d", "$\Delta$ Effect size", "Label", value_col="estimate", rot=0,
                        filename='outputs/figures/effect_size_difference_comparative')

all_results_df = pd.DataFrame.from_records(likert_results_entries+comp_results_entries, columns=["method", "label", "groups", "estimate", "CI low", "CI high"])
grouped_barplot_effectsize(all_results_df, "d", "$\Delta$ Effect size", "Label", value_col="estimate", rot=0,
                        filename='outputs/figures/effect_size_difference')




