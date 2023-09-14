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
from effect_size import effect_size_likert

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
labels = sx.xs('likert dialogue', level='category').index.unique('label').tolist()

def preprocess_likert(df_dict):
    for name, evaluation in df_dict.items():
        evaluation.columns = ['score']
        df_dict[name] = evaluation.xs('likert dialogue', level='category').reset_index().pivot(index=('bot', 'item'), columns='label', values='score').sort_index()
    return df_dict

def average_over_botpairs(bot_pairs, group1_bots, group2_bots):
    diffs = []
    for idx1, idx2 in bot_pairs:
        diff = effect_size_likert(group1_bots[idx1], group1_bots[idx2], group2_bots[idx1], group2_bots[idx2])
        diffs.append(diff)
    diffs_np = np.array(diffs)
    means = np.mean(diffs_np, axis=0)
    return means

def avg_effect_size_likert(*bot_ratings, axis=None):
    ratings_by_group = []
    for i in range(0, len(bot_ratings), 4):
        ratings_by_group.append(bot_ratings[i:i+4])
    bot_pairs = list(combinations(range(4), 2))
    botpair_strs = [(all_bots[idx1], all_bots[idx2]) for idx1, idx2 in bot_pairs]

    if len(ratings_by_group) == 4:
        # dev vs others
        dev_rating = ratings_by_group[0]
        other_ratings = ratings_by_group[1:]
        botpair_means_over_groups = []
        for other_rating in other_ratings:
            botpair_means = average_over_botpairs(bot_pairs, dev_rating, other_rating)
            botpair_means_over_groups.append(botpair_means)
        botpair_means_over_groups_np = np.array(botpair_means_over_groups)
        means = np.mean(botpair_means_over_groups_np, axis=0)
    elif len(ratings_by_group) == 3:
        # others vs others
        groupings = list(combinations(range(3), 2))
        botpair_means_over_groups = []
        for idx1, idx2 in groupings:
            botpair_means = average_over_botpairs(bot_pairs, ratings_by_group[idx1], ratings_by_group[idx2])
            botpair_means_over_groups.append(botpair_means)
        botpair_means_over_groups_np = np.array(botpair_means_over_groups)
        means = np.mean(botpair_means_over_groups_np, axis=0)
    else:
        raise Exception('Unhandled number of ratings passed to avg effect size calculation!')

    sorted_means = np.sort(means).tolist()
    if len(sorted_means) == 1:
        return sorted_means[0]
    return sorted_means

def run_bootstrap_on(df_dict, n_resamples, confidence_level, prep_func, stat_func, bots=None):
    if bots is None:
        bots = all_bots
    df_dict = prep_func(df_dict)
    group_pairs = [('dev', 'ndev', 'ui', 'sx'), ('ndev', 'ui', 'sx')]
    results = {}
    for label in labels:
        for groups in group_pairs:
            ratings_by_group = []
            for group in groups:
                group_ratings = [df_dict[group].xs(bot, level='bot')[label].tolist() for bot in bots]
                ratings_by_group.append(group_ratings)
            result = bootstrap([ratings for group_ratings in ratings_by_group for ratings in group_ratings], stat_func,
                               n_resamples=n_resamples, confidence_level=confidence_level,
                               method='bca', random_state=654)
            point_estimate = stat_func(*[ratings for group_ratings in ratings_by_group for ratings in group_ratings])
            # print(label, group1, group2, point_estimate)
            results.setdefault(label, {})[groups] = (result, point_estimate)
    return results


#####################################################################

rounds = 10000
k = 100

#####################################################################

# Novice results

print('Likert - All bots')
start = time.time()
result = run_bootstrap_on({'ui': ui, 'sx': sx, 'dev': developer, 'ndev': nondeveloper},
                          n_resamples=rounds,
                          confidence_level=0.95,
                          prep_func=preprocess_likert,
                          stat_func=avg_effect_size_likert)
end = time.time()
print(f"Elapsed: {end-start:.2f} s")


results_entries = []
for label, label_results in result.items():
    for groups, r in label_results.items():
        if 'dev' in groups:
            groups = 'Expert-Novice'
        else:
            groups = 'Novice'
        results_entries.append([label, groups, r[1], r[0].confidence_interval.low, r[0].confidence_interval.high])
results_df = pd.DataFrame.from_records(results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
grouped_barplot_cohensd(results_df, "d", "", "", ylim=(0, 1.0), value_col="estimate", rot=0,
                        filename='outputs/figures/effect_size_difference_likert_experts', plot_err=True)
