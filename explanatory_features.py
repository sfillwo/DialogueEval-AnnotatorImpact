from utilities.analysis import *
from utilities.graphing import *
import pandas as pd

from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.linear_model import OLS as LinearModel
from statsmodels.discrete.discrete_model import Logit as LogisticModel
from statsmodels.tools.tools import add_constant

import random
seed = 123
random.seed(seed)

PLOT = True

#########################################
##  DATA
#########################################

interactive_annotations = data.dialogue_collection.annotation_dataframe()
interactive_annotations_comparative = data.dialogue_collection.comparative_annotation_dataframe()

external_annotations = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123)
external_annotations_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

ux_annotations = get_singly_annotated(data_student_extcomp.student_external_comparative.annotation_dataframe(), seed=123)
ux_annotations_comparative = get_singly_annotated(data_student_extcomp.student_external_comparative.comparative_annotation_dataframe(), seed=123)


####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################




#########################################
##  PREDICTIVE VALIDITY FUNCTIONS
#########################################

def all_dialogue_metrics(data_with_behavior_labels, outcome_data):
    static: pd.DataFrame = data_with_behavior_labels.annotation_dataframe()
    static = get_singly_annotated(static, seed=123)
    reindexed = static.reset_index()
    items = reindexed[sym.item]
    dialogues = [e[0] if isinstance(e, tuple) else e for e in items]
    reindexed['dialogue'] = dialogues
    reindexed.set_index(
        [sym.bot, sym.category, sym.label, 'dialogue', sym.item],
        inplace=True, verify_integrity=True
    )
    ld = reindexed.xs(category.likert_dialogue, level=sym.category)
    ld = ld.droplevel(sym.bot).droplevel(sym.item)
    ld.columns = ['score']
    ldq = ld.xs(scale.quality, level=sym.label)
    ldq.columns = ['quality']

    lt = reindexed.xs(category.likert_turn, level=sym.category)
    lt = lt.groupby([sym.label, 'dialogue']).mean()
    lt.columns = ['score']
    ltq = lt.xs(scale.quality, level=sym.label)
    ltq.columns = ['quality']

    be = reindexed.xs(category.behavior, level=sym.category)
    be = be.groupby([sym.label, 'dialogue']).mean()
    be.columns = ['score']

    interactive = get_singly_annotated(outcome_data.annotation_dataframe(), seed=123)
    if category.likert_dialogue in interactive.index.unique('category'):
        idq = interactive.xs((category.likert_dialogue, scale.quality), level=(sym.category, sym.label))
        idq = idq.droplevel(0)
    else:
        idq = None

    ds = pd.concat(
        [lt, be, ld],
        keys=[category.likert_turn, category.behavior, category.likert_dialogue],
        names=[sym.category, sym.label, 'dialogue']
    )
    likert_dialogue_quality_features = ds.join(ldq, on='dialogue')
    likert_turn_quality_features = ds.join(ltq, on='dialogue')
    if idq is not None:
        interactive_dialogue_quality_features = ds.join(idq, on='dialogue')
        interactive_dialogue_quality_features.columns = ['score', 'quality']
    else:
        interactive_dialogue_quality_features = None

    icq, scq = comparative_dialogue_metrics(ds, data_with_behavior_labels, outcome_data)

    return (
        likert_dialogue_quality_features,
        likert_turn_quality_features,
        icq,
        scq,
        interactive_dialogue_quality_features
    )

def comparative_dialogue_metrics(ds, data_with_behavior_labels, outcome_data):
    def create_comparative_feature_data(comparative_df):
        interactive_comparisons = comparative_df
        surge_comparisons = get_singly_annotated(data_with_behavior_labels.comparative_annotation_dataframe(), seed=123)
        compared_dialogues = surge_comparisons.index.get_level_values('dialogues')
        unique_compared_dialogues = {tuple(x) for x in {frozenset(y) for y in compared_dialogues}}
        comparison_map = dict(unique_compared_dialogues)
        compared_selector = [
            pair in unique_compared_dialogues
            for pair in interactive_comparisons.index.get_level_values('dialogues')
        ]
        comparative: pd.DataFrame = interactive_comparisons.loc[compared_selector, :]
        compared_selector = [
            pair in unique_compared_dialogues
            for pair in surge_comparisons.index.get_level_values('dialogues')
        ]
        surge_comparisons: pd.DataFrame = surge_comparisons.loc[compared_selector, :]
        comparative_quality = comparative.xs(scale.quality, level=sym.label)
        comparative_quality.index = [first for _, _, (first, second) in comparative_quality.index.values]
        comparative_quality.columns = ['quality']
        surge_comparisons.index = pd.MultiIndex.from_arrays(
            list(zip(*[
                (category.comparative, label, left)
                for _, _, label, (left, right) in surge_comparisons.index.values
            ])),
            names=[sym.category, sym.label, 'dialogue']
        )
        surge_comparisons.columns = ['score']
        filtered_ds = ds.loc[[(c, l, d) for c, l, d in ds.index.values if d in comparison_map]]
        compared_features = ds.loc[[(c, l, comparison_map[d]) for c, l, d in filtered_ds.index.values]]
        comparative_features = filtered_ds.to_numpy() - compared_features.to_numpy()
        filtered_ds['diff'] = comparative_features.squeeze().tolist()
        del filtered_ds['score']
        filtered_ds.columns = ['score']
        filtered_ds = pd.concat([filtered_ds, surge_comparisons], axis=0)
        comparative_quality_features = filtered_ds.join(comparative_quality, on='dialogue')
        icq = comparative_quality_features
        icq = icq[icq['quality'] != 0]
        icq.loc[:, 'quality'] = (icq['quality'] > 0).astype(int)
        return icq

    icq = create_comparative_feature_data(
        get_singly_annotated(outcome_data.comparative_annotation_dataframe(), seed=123))
    scq = create_comparative_feature_data(
        get_singly_annotated(data_with_behavior_labels.comparative_annotation_dataframe(), seed=123))

    return icq, scq


def regressions(df, quality_column_name=None, model='linear', adjust_r2=False):
    """
    :param df: dialogue x (*features, quality) -> value
    :return: *(coef, low, high), mcfadden r^2
    """
    if not quality_column_name:
        quality_column_name = df.columns[-1]
    qualities = df[quality_column_name]
    features = [f for f in df.columns if f != quality_column_name]
    if model == 'ordinal':
        model = OrderedModel(qualities, df[features], distr='logit')
        results = model.fit()
        coefs = {f: results.params[f] for f in features}
        prsqrd = results.prsquared
        if adjust_r2:
            prsqrd = 1 - (results.llf - len(features)) / results.llnull
        result = {stat.mcfad_r2: prsqrd, stat.p_of_llr_test: results.llr_pvalue}
    elif model == 'linear':
        x = add_constant(df[features])
        y = qualities
        model = LinearModel(y, x)
        results = model.fit()
        coefs = {f: results.params[f] for f in features}
        rsquared = results.rsquared
        if adjust_r2:
            adjust = lambda r2, f: 1 - (1 - r2) * ((len(y) - 1) / (len(y) - f - 1))
            rsquared = adjust(rsquared, len(features))
        result = {**coefs, stat.r2: rsquared, stat.p_of_f_test: results.f_pvalue}
    elif model == 'logistic':
        x = add_constant(df[features])
        y = qualities.astype(bool)
        model = LogisticModel(y, x)
        results = model.fit()
        coefs = {f: results.params[f] for f in features}
        prsqrd = results.prsquared
        if adjust_r2:
            prsqrd = 1 - (results.llf - len(features)) / results.llnull
        result = {**coefs, stat.mcfad_r2: prsqrd, stat.p_of_llr_test: results.llr_pvalue}
    else:
        raise ValueError('Param "model" must be one of {"linear", "ordinal", "logistic"}')
    return pd.Series(result.values(), result)

@to_file
def dialogue_quality_regressions(external_data, interactive_data):
    ldq, ltq, icq, scq, idq = all_dialogue_metrics(external_data, interactive_data)
    ldq_groups = ldq.groupby([sym.category, sym.label])
    ltq_groups = ltq.groupby([sym.category, sym.label])
    scq_groups = scq.groupby([sym.category, sym.label])
    scq_noqual = scq.drop(scq_groups.get_group((category.comparative, scale.quality)).index)
    scq_groups = scq_noqual.groupby([sym.category, sym.label])
    icq_groups = icq.groupby([sym.category, sym.label])
    if idq is not None:
        idq_groups = idq.groupby([sym.category, sym.label])
    names = ['Predicted', 'Metric']
    linear_compare_result = icq_groups.apply(lambda x: regressions(x, model='logistic'))
    linear_compare_result.columns = pd.MultiIndex.from_arrays(
        [['Interactive Comparison']*3,
        ['LC Coefficient', 'LC Pseudo R-Squared', stat.p_of_llr_test]],
        names=names
    )
    static_compare_result = scq_groups.apply(lambda x: regressions(x, model='logistic'))
    static_compare_result.columns = pd.MultiIndex.from_arrays(
        [['Static Comparison']*3,
        ['LC Coefficient', 'LC Pseudo R-Squared', stat.p_of_llr_test]],
        names=names
    )
    linear_result = ldq_groups.apply(lambda x: regressions(x, model='linear'))
    linear_result.columns = pd.MultiIndex.from_arrays(
        [['Likert Dialogue Quality']*3,
        ['LR Coefficient', 'LR R-Squared', stat.p_of_f_test]],
        names=names
    )
    ordinal_result = ldq_groups.apply(lambda x: regressions(x, model='ordinal'))
    ordinal_result.columns = pd.MultiIndex.from_arrays(
        [['Likert Dialogue Quality']*2,
        ['OR Pseudo R-Squared', stat.p_of_llr_test]],
        names=names
    )
    linear_turn_result = ltq_groups.apply(regressions)
    linear_turn_result.columns = pd.MultiIndex.from_arrays(
        [['Likert Turn Quality']*3,
        ['LR Coefficient', 'LR R-Squared', stat.p_of_f_test]],
        names=names
    )
    if idq is not None:
        interactive_dial_result = idq_groups.apply(regressions)
        interactive_dial_result.columns = pd.MultiIndex.from_arrays(
            [['Interactive Likert']*3,
            ['LR Coefficient', 'LR R-Squared', stat.p_of_f_test]],
            names=names
        )
        result = pd.concat(( linear_compare_result, static_compare_result, interactive_dial_result, linear_result, linear_turn_result), axis=1)
    else:
        result = pd.concat(( linear_compare_result, static_compare_result, linear_result, linear_turn_result), axis=1)
    return result.round(5)

def get_rsquareds(data, filename, datas, iv, reload=True):

    if reload:
        regs = dialogue_quality_regressions(
            *datas,
            reload=filename
        )
    else:
        regs = dialogue_quality_regressions(
            *datas,
            load=filename
        )

    qual_abbrev = {
        'Interactive Comparison': 'I Qua$_c$',
        'Interactive Likert': 'I Qua$_d$',
        'Likert Dialogue Quality': 'X Qua$_d$',
        'Static Comparison': 'X Qua$_c$'
    }

    just_rsquareds = regs[iv]
    just_rsquareds = just_rsquareds.drop(("likert dialogue", "quality"))
    flattened_index = just_rsquareds.index.to_flat_index()
    order = ['behavior', 'likert turn'] #'likert dialogue', 'comparative'
    reordered_flattened_index = [x for c in order for x in flattened_index if c in x]
    reordered_multindex = pd.MultiIndex.from_tuples(reordered_flattened_index)
    just_rsquareds = just_rsquareds.reindex(reordered_multindex)
    just_rsquareds.columns = [qual_abbrev[col[0]] for col in just_rsquareds.columns]
    just_rsquareds.drop('quality', level=1, inplace=True)
    return just_rsquareds

#########################################
##  PREDICTIVE VALIDITY - EXTERNAL RESULTS
#########################################
datas = [data.surge_evaluation, data.dialogue_collection]

external_rsquareds = get_rsquareds(data, filename='outputs/results/dialogue_quality_regressions_external',
                                   iv=[("Static Comparison", "LC Pseudo R-Squared"),
                                       ("Likert Dialogue Quality", "LR R-Squared")],
                                   datas=datas, reload=False)
# external_rsquareds.columns = ['X Qua$_c$', 'X Qua$_d$']
plot_predictive_validity(external_rsquareds)

#########################################
##  PREDICTIVE VALIDITY - INTERACTIVE RESULTS
#########################################

interactive_rsquareds = get_rsquareds(data, filename='outputs/results/dialogue_quality_regressions_interactive',
                                      iv=[("Interactive Comparison", "LC Pseudo R-Squared"),
                                          ("Interactive Likert", "LR R-Squared")],
                                      datas=datas, reload=False)
# interactive_rsquareds.columns = ['I Qua$_c$', 'I Qua$_d$']
plot_predictive_validity(interactive_rsquareds)

#########################################
##  PREDICTIVE VALIDITY - Ux RESULTS
#########################################

datas = [data.surge_evaluation, data_student_extcomp.student_external_comparative]

ux_rsquareds = get_rsquareds(data, filename='outputs/results/dialogue_quality_regressions_ux',
                             iv=[("Interactive Comparison", "LC Pseudo R-Squared")],
                             datas=datas, reload=False)
# ux_rsquareds.columns = ['X Qua$_c$', 'X Qua$_d$']
plot_predictive_validity(ux_rsquareds)

#########################################
##  PREDICTIVE VALIDITY - COMBINED RESULTS
#########################################

likert_rsquareds = pd.concat([interactive_rsquareds['I Qua$_d$'], external_rsquareds['X Qua$_d$']], axis=1)
likert_rsquareds.columns = ['Ui Qua$_d$', 'Sx Qua$_d$']
likert_rsquareds.index.names = ['category', 'label']
plot_predictive_validity(likert_rsquareds, sorted=['category', 'Ui Qua$_d$'])

comparative_rsquareds = pd.concat([interactive_rsquareds['I Qua$_c$'], ux_rsquareds['I Qua$_c$'], external_rsquareds['X Qua$_c$']], axis=1)
comparative_rsquareds.columns = ['Ui Qua$_c$', 'Ux Qua$_c$', 'Sx Qua$_c$']
comparative_rsquareds.index.names = ['category', 'label']
plot_predictive_validity(comparative_rsquareds, sorted=['category', 'Ui Qua$_c$'])


####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################




#########################################
##  INCREMENTAL VALIDITY FUNCTIONS
#########################################

# def drop_column_level_duplication(df: pd.DataFrame, columns, levels=None):
#     if levels is None:
#         levels = list(range(len(columns)))
#     level_columns = df.xs(columns, axis=1, level=levels)
#     unique = level_columns.iloc[:,0].to_frame()
#     unique.columns = [columns]
#     dropped = df.drop(columns=columns, level=levels)
#     result = pd.concat([dropped, unique], axis=1)
#     return result
#
# def multivariate_regression(df: pd.DataFrame, model='linear', adjust_r2=True):
#     def apply_regressions(df: pd.DataFrame):
#         unstacked = df.unstack([sym.category, sym.label])
#         unstacked = drop_column_level_duplication(unstacked, 'quality', 0)
#         results = regressions(unstacked, quality_column_name='quality', model=model, adjust_r2=adjust_r2)
#         return results
#     result = apply_regressions(df)
#     result.index = [
#         (idx[1] if isinstance(idx, tuple) else idx)
#         for idx in result.index.values
#     ]
#     return result.round(5)
#
# from collections import namedtuple
# from matplotlib.patches import Patch
#
# @to_file
# def incremental_regression(
#         df: pd.DataFrame,
#         categories,
#         model='linear',
#         beam=1,
#         select='backward',
#         exclusions=(),
#         adjust_r2=True
# ):
#     data_points = set(df.index.get_level_values('dialogue'))
#     num_data_points = len(data_points)
#     Step: type = namedtuple('Step', ('r2', 'p', 'feature'))
#     class Path(list):
#         def metric(self):
#             return self.r2
#         @property
#         def r2(self):
#             return self[-1].r2 if self else 0
#         @property
#         def p(self): return self[-1].p if self else 1
#         @property
#         def features(self): return {x.feature for x in self}
#     r2_name = stat.r2 if model=='linear' else stat.mcfad_r2
#     p_name = stat.p_of_f_test if model=='linear' else stat.p_of_llr_test
#     frontier = [Path()]
#     feature_pool = {
#         x[:2] for x in df.index.values
#         if (not (x in exclusions or x[1] in exclusions))
#         and x[0] in categories
#     }
#     for _ in feature_pool:
#         new_frontier = []
#         for path in frontier:
#             for candidate in feature_pool - path.features:
#                 if select == 'forward':
#                     candidate_features = path.features | {candidate}
#                 elif select == 'backward':
#                     candidate_features = feature_pool - path.features
#                 else:
#                     raise ValueError('param select must be one of {"forward", "backward"}')
#                 row_mask = [
#                     x[:2] in candidate_features
#                     and (not (x in exclusions or x[1] in exclusions))
#                     and x[0] in categories
#                     for x in df.index.values
#                 ]
#                 candidate_df = df.loc[row_mask, :]
#                 candidate_results = multivariate_regression(candidate_df, model=model, adjust_r2=adjust_r2)
#                 r2 = candidate_results[r2_name].item()
#                 p = candidate_results[p_name]
#                 new_frontier.append(Path([*path, Step(r2, p, candidate)]))
#         frontier = sorted(new_frontier, key=lambda x: x.metric(), reverse=True)[:beam]
#     result = {
#         step.feature: {r2_name: step.r2, p_name: step.p}
#         for i, step in enumerate(frontier[0])
#     }
#     return pd.DataFrame(result.values(), result)
#
#
# ldq, ltq, icq, scq, idq = all_dialogue_metrics(data.surge_evaluation, data.dialogue_collection)
#
#
# DO_ADJUST = True
#
#
# behavior_sldq = incremental_regression(
#     ldq, (category.behavior,), beam=10, adjust_r2=DO_ADJUST,
#     load=f'outputs/results/behavior_incremental_regressions_adjusted={DO_ADJUST}'
# )
#
# behavior_scq = incremental_regression(
#     scq, (category.behavior,), beam=10, model='logistic', adjust_r2=DO_ADJUST,
#     load=f'outputs/results/behavior_incremental_regressions_comparative_adjusted={DO_ADJUST}'
# )
#
# behavior_ildq = incremental_regression(
#     idq, (category.behavior,), beam=10, adjust_r2=DO_ADJUST,
#     load=f'outputs/results/behavior_incremental_regressions_interactive_adjusted={DO_ADJUST}'
# )
#
# behavior_icq = incremental_regression(
#     icq, (category.behavior,), beam=10, model='logistic', adjust_r2=DO_ADJUST,
#     load=f'outputs/results/behavior_incremental_regressions_comparative_interactive_adjusted={DO_ADJUST}'
# )
#
# turn_sldq = incremental_regression(
#     ldq, (category.likert_turn,), beam=10, exclusions=[scale.quality], adjust_r2=DO_ADJUST,
#     load=f'outputs/results/likert_turn_incremental_regressions_adjusted={DO_ADJUST}'
# )
#
# turn_scq = incremental_regression(
#     scq, (category.likert_turn,), beam=10, model='logistic', exclusions=['quality'], adjust_r2=DO_ADJUST,
#     load=f'outputs/results/likert_turn_incremental_regressions_comparative_adjusted={DO_ADJUST}'
# )
#
# turn_ildq = incremental_regression(
#     idq, (category.likert_turn,), beam=10, exclusions=[scale.quality], adjust_r2=DO_ADJUST,
#     load=f'outputs/results/likert_turn_incremental_regressions_interactive_adjusted={DO_ADJUST}'
# )
#
# turn_icq = incremental_regression(
#     icq, (category.likert_turn,), beam=10, model='logistic', exclusions=['quality'], adjust_r2=DO_ADJUST,
#     load=f'outputs/results/likert_turn_incremental_regressions_comparative_interactive_adjusted={DO_ADJUST}'
# )
#
# # dialogue_sldq = incremental_regression(
# #     ldq, (category.likert_dialogue,), beam=10, exclusions=['quality'], adjust_r2=DO_ADJUST,
# #     load='outputs/results/likert_dialogue_incremental_regressions'
# # )
# #
# # dialogue_scq = incremental_regression(
# #     scq, (category.likert_dialogue,), beam=10, model='logistic', exclusions=['quality'], adjust_r2=DO_ADJUST,
# #     load='outputs/results/likert_dialogue_incremental_regressions_comparative'
# # )
# #
# # dialogue_ildq = incremental_regression(
# #     idq, (category.likert_dialogue,), beam=10, exclusions=['quality'], adjust_r2=DO_ADJUST,
# #     load='outputs/results/likert_dialogue_incremental_regressions_interactive'
# # )
# #
# # dialogue_icq = incremental_regression(
# #     icq, (category.likert_dialogue,), beam=10, model='logistic', exclusions=['quality'], adjust_r2=DO_ADJUST,
# #     load='outputs/results/likert_dialogue_incremental_regressions_comparative_interactive'
# # )
# #
# # comparative_icq = incremental_regression(
# #     icq, (category.comparative,), beam=10, model='logistic', exclusions=[scale.quality], adjust_r2=DO_ADJUST,
# #     load='outputs/results/comparative_incremental_regressions_comparative'
# # )
#
# cutoff_start = { # label for which it and all subsequent labels are monotonically decreasing in adjusted r-squared from previous label
#     # "comparative_icq": "informative",
#     # "dialogue_icq_ex": "informative",
#     # "dialogue_icq": "emotional",
#     # "dialogue_ildq": "engaging",
#     # "dialogue_scq": "informative",
#     # "turn_icq_ex": "grammatical",
#     # "turn_icq": "grammatical",
#     # "turn_ildq": "informative",
#     # "turn_sldq": "informative",
#     # "turn_scq": "relevant",
#     # "behavior_icq": "follow up",
#     # "behavior_ildq": "follow up",
#     # "behavior_sldq": "ignore",
#     # "behavior_scq": "partner contradiction",
# }
#
#
# # Build the plot
# SMALL_SIZE = 16
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=18)            # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#
# plt.rcParams["figure.figsize"] = (13,13)
#
# fig, ax = plt.subplots()
#
# def plot_by_category(ax, df, category, style, color, xaxis_start, annot_pos_ls, rotations=None):
#     extracted = df.reset_index()[::-1]
#     subscript = category.split("_")[0][0]
#     labels = [dimensions_transformer[l]+f'$_{subscript}$' for l in extracted["level_1"]]
#     xaxis_end = xaxis_start + len(extracted)
#     column = [x for x in extracted.columns if 'r-squared' in x.lower()][0]
#     xs = np.arange(xaxis_start, xaxis_end)
#     ys = extracted[column]
#     if rotations is None:
#         rotations = [0]*len(xs)
#     adjusted_cutoff = cutoff_start.get(category, None)
#     if adjusted_cutoff:
#         adjusted_cutoff_idx = extracted["level_1"].tolist().index(adjusted_cutoff)
#     else:
#         adjusted_cutoff_idx = len(xs)
#     ax.plot(xs[:adjusted_cutoff_idx],
#             ys[:adjusted_cutoff_idx],
#             marker='o',
#             linestyle=style,
#             linewidth=5,
#             color=color)
#     ax.plot(xs[adjusted_cutoff_idx-1:],
#             ys[adjusted_cutoff_idx-1:],
#             marker='o',
#             linestyle=style,
#             linewidth=2,
#             color=color)
#     for x,y,l,annot_pos,rot in zip(xs, ys, labels,annot_pos_ls, rotations):
#         plt.annotate(l,
#                      (x,y),
#                      textcoords="offset points",
#                      xytext=annot_pos,
#                      ha='center',
#                      fontsize=15,
#                      rotation=rot)
#     return labels, xaxis_end
#
# external_likert_color = "blue"
# external_comparative_color = "green"
# interactive_likert_color = "orange"
# interactive_comparative_color = "red"
#
# behavior_line = 'solid'
# likert_turn_line = 'dotted'
#
# if DO_ADJUST:
#     # interactive comparative quality
#     behavior_labels, _ = plot_by_category(ax, behavior_icq, "behavior_icq", behavior_line, interactive_comparative_color, 1,
#                                           [(0, -20)] + [(20, -10)] * 7 + [(10, -15)] + [(0, -15)] * 7)
#     likert_turn_labels, _ = plot_by_category(ax, turn_icq, "turn_icq", likert_turn_line, interactive_comparative_color, 16,
#                                              [(-15, 10)] * 3 + [(0, 10)] * 5)
#
#     # interactive likert quality
#     _, _ = plot_by_category(ax, behavior_ildq, "behavior_ildq", behavior_line, interactive_likert_color, 0,
#                             [(0, -20)] + [(-20, 0)] * 5 + [(0, 10)] * 10)
#     _, _ = plot_by_category(ax, turn_ildq, "turn_ildq", likert_turn_line, interactive_likert_color, 14,
#                             [(0, -20)] + [(15, -15)] * 3 + [(5, -20)] + [(0, -20)] * 2)
#
#     # surge comparative quality
#     _, _ = plot_by_category(ax, behavior_scq, "behavior_scq", behavior_line, external_comparative_color, 0,
#                             [(0, -20)] + [(20, -10)] * 4 + [(5, -20)] * 3 + [(5, -25)] + [(0, -30)] * 7)
#     _, _ = plot_by_category(ax, turn_scq, "turn_scq", likert_turn_line, external_comparative_color, 18, [(0, 10)] * 8)
#
#     # surge likert quality
#     _, _ = plot_by_category(ax, behavior_sldq, "behavior_sldq", behavior_line, external_likert_color, 1, [(0, 10)] * 16)
#     _, _ = plot_by_category(ax, turn_sldq, "turn_sldq", likert_turn_line, external_likert_color, 18, [(0, -20)] * 8)
# else:
#     # interactive comparative quality
#     behavior_labels, _ = plot_by_category(ax, behavior_icq, "behavior_icq", behavior_line, interactive_comparative_color, 2,
#                                           [(0, -20)] + [(20,-10)]*7 + [(10,-15)] + [(0,-15)]*7)
#     likert_turn_labels, _ = plot_by_category(ax, turn_icq, "turn_icq", likert_turn_line, interactive_comparative_color, 16,
#                                              [(-15,10)]*3 + [(0,10)]*5)
#
#     # interactive likert quality
#     _, _ = plot_by_category(ax, behavior_ildq, "behavior_ildq", behavior_line, interactive_likert_color, 0,
#                             [(0, -20)] + [(-20,0)]*5 + [(0,10)]*10)
#     _, _ = plot_by_category(ax, turn_ildq, "turn_ildq", likert_turn_line, interactive_likert_color, 12,
#                             [(0,-20)] + [(15,-15)]*3 + [(5,-20)] + [(0,-20)]*2)
#
#     # surge comparative quality
#     _, _ = plot_by_category(ax, behavior_scq, "behavior_scq", behavior_line, external_comparative_color, 1,
#                             [(0, -20)] + [(20,-10)]*4 + [(5,-20)]*3 + [(5,-25)] + [(0,-30)]*7,
#                             [0]*5 + [-15]*3 + [-45]*8)
#     _, _ = plot_by_category(ax, turn_scq, "turn_scq", likert_turn_line, external_comparative_color, 6, [(0,10)]*8)
#
#     # surge likert quality
#     _, _ = plot_by_category(ax, behavior_sldq, "behavior_sldq", behavior_line, external_likert_color, 1, [(0,10)]*16)
#     _, _ = plot_by_category(ax, turn_sldq, "turn_sldq", likert_turn_line, external_likert_color, 5, [(0,-20)]*8)
#
# ax.set_ylabel(r"$R^2$", labelpad=20, rotation=0)
# ax.set_xticks([])
# ax.yaxis.grid(True)
#
# from matplotlib.lines import Line2D
# custom_lines = [
#     Line2D([0], [0], linestyle=behavior_line, color='black', lw=5),
#     Line2D([0], [0], linestyle=likert_turn_line, color='black', lw=5),
#     Patch(facecolor=interactive_likert_color),
#     Patch(facecolor=interactive_comparative_color),
#     Patch(facecolor=external_likert_color),
#     Patch(facecolor=external_comparative_color)
# ]
# ax.legend(custom_lines,
#           ['ABC-Eval', 'Likert Turn', 'Interactive Qua$_d$', 'Interactive Qua$_c$', 'External Qua$_d$','External Qua$_c$'],
#           # ['Interactive Qua$_c$', 'Interactive Qua$_d$'],
#           handlelength=4,
#           loc='upper right')
#
# # Save the figure and show
# plt.tight_layout()
# plt.show()
