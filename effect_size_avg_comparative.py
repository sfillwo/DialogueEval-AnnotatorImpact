import itertools

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

from utilities.analysis import *
from utilities.mappings import bot_transformer
from utilities.graphing import grouped_barplot_benchmark, grouped_barplot_cohensd

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

def preprocess_comparative(evaluation):
    evaluation.columns = ['score']
    return evaluation.xs('comparative', level='category').reset_index().pivot(index=('bot', 'item'), columns='label', values='score').sort_index()

def avg_cohensh(d1, d2, d3, d4, axis=None):
    d1d2 = cohensh(d1, d2)
    d1d3 = cohensh(d1, d3)
    d1d4 = cohensh(d1, d4)
    d2d3 = cohensh(d2, d3)
    d2d4 = cohensh(d2, d4)
    d3d4 = cohensh(d3, d4)
    return np.add.reduce([d1d2, d1d3, d1d4, d2d3, d2d4, d3d4]) / 6

def win_proportion(arr):
    wins = np.count_nonzero(arr == 1, axis=1)
    losses = np.count_nonzero(arr == -1, axis=1)
    win_props = wins / (wins + losses)
    return win_props

def cohensh(d1, d2):
    if isinstance(d1, list):
        d1 = np.array(d1)
    if isinstance(d2, list):
        d2 = np.array(d2)
    if len(d1.shape) == 1:
        n1 = d1.shape[0]
        d1 = d1.reshape((1, -1))
    elif len(d1.shape) == 2:
        n1 = d1.shape[1]
    if len(d2.shape) == 1:
        n2 = d2.shape[0]
        d2 = d2.reshape((1, -1))
    elif len(d2.shape) == 2:
        n2 = d2.shape[1]
    p1 = win_proportion(d1)
    p2 = win_proportion(d2)
    return np.abs(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2)))

def run_bootstrap_on(df, n_resamples, confidence_level, prep_func, stat_func, bots=None):
    if bots is None:
        bots = all_bots
    df = prep_func(df)
    results = {}
    for label in labels:
        ratings = [df.xs(bot, level='bot')[label].tolist() for bot in bots]
        result = bootstrap(ratings, stat_func,
                           n_resamples=n_resamples, confidence_level=confidence_level,
                           method='percentile', random_state=654)
        point_estimate = stat_func(*ratings)
        results[label] = (result, point_estimate[0])
    return results


#####################################################################

rounds = 10000
k = 100

#####################################################################

start = time.time()
result_ui = run_bootstrap_on(ui,
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_cohensh)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")

start = time.time()
result_sx = run_bootstrap_on(sx,
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_cohensh)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")

start = time.time()
result_dev = run_bootstrap_on(developer,
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_cohensh)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")

start = time.time()
result_ndev = run_bootstrap_on(nondeveloper,
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_comparative,
                          stat_func=avg_cohensh)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")

results_entries = []
for label, r in result_ui.items():
    results_entries.append([label, 'Stu$_i$', r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
for label, r in result_sx.items():
    results_entries.append([label, 'Sur$_x$', r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
for label, r in result_dev.items():
    results_entries.append([label, 'Dev$_x$', r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
for label, r in result_ndev.items():
    results_entries.append([label, 'Stu$_x$', r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])

results_df = pd.DataFrame.from_records(results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])

pivoted_results_df = results_df.pivot(
    index="label",
    columns="groups",
    values="estimate"
)
pivoted_results_df = pivoted_results_df.round(decimals=2)
print(pivoted_results_df)

# grouped_barplot_cohensd(results_df, None, "", "", value_col="estimate",
#                         filename="outputs/figures/effect_sizes_per_group_comparative",
#                         ylim=(0,1.25), fig_size=(10,3), rot=0, width=0.8)


