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
    sorted_means = np.sort(means).tolist()
    if len(sorted_means) == 1:
        return sorted_means[0]
    return sorted_means

def run_bootstrap_on(df_dict, n_resamples, confidence_level, prep_func, stat_func, bots=None):
    if bots is None:
        bots = all_bots
    df_dict = prep_func(df_dict)
    group_pairs = [('ui', 'sx'), ('dev', 'ui'), ('dev', 'sx'), ('dev', 'ndev'), ('ui', 'ndev'), ('ndev', 'sx')]
    results = {}
    for label in labels:
        for group1, group2 in group_pairs:
            group1_ratings = [df_dict[group1].xs(bot, level='bot')[label].tolist() for bot in bots]
            group2_ratings = [df_dict[group2].xs(bot, level='bot')[label].tolist() for bot in bots]
            result = bootstrap([*group1_ratings, *group2_ratings], stat_func,
                               n_resamples=n_resamples, confidence_level=confidence_level,
                               method='bca', random_state=654)
            point_estimate = stat_func(*group1_ratings, *group2_ratings)
            # print(label, group1, group2, point_estimate)
            results.setdefault(label, {})[(group1, group2)] = (result, point_estimate)
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

group_mapping = {
    'ui': 'Stu$_i$',
    'sx': 'Sur$_x$',
    'dev': 'Dev$_x$',
    'ndev': 'Stu$_x$',
}

# group_ordering = {
#     "4|Stu$_i$/Sur$_x$": ui_sx_agreements,
#     "7|Dev$_x$/Stu$_i$": dev_ui_agreements,
#     "8|Dev$_x$/Sur$_x$": dev_sx_agreements,
#     "3|Dev$_x$": dev_agreements,
#     "5|Stu$_x$/Stu$_i$": ndev_ui_agreements,
#     "6|Stu$_x$/Sur$_x$": ndev_sx_agreements,
#     "1|Stu$_x$": ndev_agreements,
#     "9|Stu$_x$/Dev$_x$": ndev_dev_agreements,
#     "2|Sur$_x$": sx_sx_agreements
# }

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
                        filename='outputs/figures/effect_size_difference_likert', plot_err=True,
                        fig_size=(20,3), width=0.8)


