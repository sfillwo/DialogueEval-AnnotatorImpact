from utilities.analysis import *
from utilities.utils import *
from utilities.graphing import *
import pandas as pd

PLOT = False

#########################################
##  DATA
#########################################

# SX
external_annotations = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123)
external_annotations_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

# UI
interactive_annotations = data.dialogue_collection.annotation_dataframe()
interactive_annotations_comparative = data.dialogue_collection.comparative_annotation_dataframe()

# Developers
developer = get_singly_annotated(developer_ext.student_external.annotation_dataframe(), seed=123)
developer_comparative = get_singly_annotated(developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

# Non developers
nondeveloper = get_singly_annotated(non_developer_ext.student_external.annotation_dataframe(), seed=123)
nondeveloper_comparative = get_singly_annotated(non_developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

#########################################
##  Dev V Dev AGREEMENTS
#########################################

dev_agreements = agreement_dataframe(
    get_doubly_annotated(
        developer_ext.student_external.annotation_dataframe()
    ),
    load='outputs/results/dev_krippendorf'
)

#########################################
##  Dev V UI AGREEMENTS
#########################################

dev_ui = pd.concat([developer, interactive_annotations], axis=1).dropna()
dev_ui.columns = ['dev', 'ui']

dev_ui_agreements = agreement_dataframe(
    dev_ui, load='outputs/results/dev_ui_krippendorf'
)

#########################################
##  Dev V SX AGREEMENTS
#########################################

dev_sx = pd.concat([developer, external_annotations], axis=1).dropna()
dev_sx.columns = ['dev', 'sx']

dev_sx_agreements = agreement_dataframe(
    dev_sx, load='outputs/results/dev_sx_krippendorf'
)

#########################################
##  NDev V NDev AGREEMENTS
#########################################

ndev_agreements = agreement_dataframe(
    get_doubly_annotated(
        non_developer_ext.student_external.annotation_dataframe()
    ),
    load='outputs/results/ndev_krippendorf'
)

#########################################
##  Dev V UI AGREEMENTS
#########################################

ndev_ui = pd.concat([nondeveloper, interactive_annotations], axis=1).dropna()
ndev_ui.columns = ['ndev', 'ui']

ndev_ui_agreements = agreement_dataframe(
    ndev_ui, load='outputs/results/ndev_ui_krippendorf'
)

#########################################
##  Dev V SX AGREEMENTS
#########################################

ndev_sx = pd.concat([nondeveloper, external_annotations], axis=1).dropna()
ndev_sx.columns = ['ndev', 'sx']

ndev_sx_agreements = agreement_dataframe(
    ndev_sx, load='outputs/results/ndev_sx_krippendorf'
)

#########################################
##  NDev V Dev AGREEMENTS
#########################################

ndev_dev = pd.concat([nondeveloper, developer], axis=1).dropna()
ndev_dev_agreements = agreement_dataframe(
    ndev_dev,
    load='outputs/results/ndev_dev_krippendorf'
)

#########################################
##  UI V SX AGREEMENTS
#########################################

ui_sx = extract_just_likert_comparative(external_annotations, interactive_annotations)

ui_sx_agreements = agreement_dataframe(
    ui_sx, load='outputs/results/ui_sx_krippendorf'
)

#########################################
##  SX V SX AGREEMENTS
#########################################

sx_annotations = data.surge_evaluation.annotation_dataframe()
likert_sx = sx_annotations.xs('likert dialogue', level=1, drop_level=False)
comparative_sx = sx_annotations.xs('comparative', level=1, drop_level=False)
sx_sx = pd.concat([likert_sx, comparative_sx])

sx_sx_agreements = agreement_dataframe(
    sx_sx, load='outputs/results/sx_sx_krippendorf'
)

#########################################
##  Table
#########################################

results_df = pd.DataFrame.from_dict({
    'ui_sx_l': ui_sx_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'ui_sx_l-': ui_sx_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'ui_sx_l+': ui_sx_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'dev_ui_l': dev_ui_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_ui_l-': dev_ui_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'dev_ui_l+': dev_ui_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'dev_sx_l': dev_sx_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_sx_l-': dev_sx_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'dev_sx_l+': dev_sx_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'dev_dev_l': dev_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_dev_l-': dev_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'dev_dev_l+': dev_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'ndev_ui_l': ndev_ui_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_ui_l-': ndev_ui_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'ndev_ui_l+': ndev_ui_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'ndev_sx_l': ndev_sx_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_sx_l-': ndev_sx_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'ndev_sx_l+': ndev_sx_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'ndev_ndev_l': ndev_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_ndev_l-': ndev_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'ndev_ndev_l+': ndev_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'ndev_dev_l': ndev_dev_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_dev_l-': ndev_dev_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'ndev_dev_l+': ndev_dev_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'sx_sx_l': sx_sx_agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict(),
    'sx_sx_l-': sx_sx_agreements.xs('likert dialogue', level='category')["CI low"].to_dict(),
    'sx_sx_l+': sx_sx_agreements.xs('likert dialogue', level='category')["CI high"].to_dict(),

    'ui_sx_c': ui_sx_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'ui_sx_c-': ui_sx_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'ui_sx_c+': ui_sx_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'dev_ui_c': dev_ui_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_ui_c-': dev_ui_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'dev_ui_c+': dev_ui_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'dev_sx_c': dev_sx_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_sx_c-': dev_sx_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'dev_sx_c+': dev_sx_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'dev_dev_c': dev_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'dev_dev_c-': dev_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'dev_dev_c+': dev_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'ndev_ui_c': ndev_ui_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_ui_c-': ndev_ui_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'ndev_ui_c+': ndev_ui_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'ndev_sx_c': ndev_sx_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_sx_c-': ndev_sx_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'ndev_sx_c+': ndev_sx_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'ndev_ndev_c': ndev_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_ndev_c-': ndev_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'ndev_ndev_c+': ndev_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'ndev_dev_c': ndev_dev_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'ndev_dev_c-': ndev_dev_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'ndev_dev_c+': ndev_dev_agreements.xs('comparative', level='category')["CI high"].to_dict(),

    'sx_sx_c': sx_sx_agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict(),
    'sx_sx_c-': sx_sx_agreements.xs('comparative', level='category')["CI low"].to_dict(),
    'sx_sx_c+': sx_sx_agreements.xs('comparative', level='category')["CI high"].to_dict(),
})

results_df.to_csv('outputs/csv/convo_agreement.csv', float_format="%.2f")

pd.set_option('display.max_columns', None)
pd.set_option("max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(results_df)

result_dict = {
    "8|Stu$_i$/Sur$_x$": ui_sx_agreements,
    "4|Dev$_x$/Stu$_i$": dev_ui_agreements,
    "6|Dev$_x$/Sur$_x$": dev_sx_agreements,
    "3|Dev$_x$": dev_agreements,
    "7|Stu$_i$/Stu$_x$": ndev_ui_agreements,
    "9|Stu$_x$/Sur$_x$": ndev_sx_agreements,
    "1|Stu$_x$": ndev_agreements,
    "5|Dev$_x$/Stu$_x$": ndev_dev_agreements,
    "2|Sur$_x$": sx_sx_agreements
}

# Full

likert_results_entries, comparative_results_entries = [], []
for grouping, agreements in result_dict.items():
    if True: #'Dev' not in grouping:
        likert = agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict()
        likert_h = agreements.xs('likert dialogue', level='category')["CI low"].to_dict()
        likert_l = agreements.xs('likert dialogue', level='category')["CI high"].to_dict()

        comparative = agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict()
        comparative_h = agreements.xs('comparative', level='category')["CI low"].to_dict()
        comparative_l = agreements.xs('comparative', level='category')["CI high"].to_dict()

        for label, score in likert.items():
            likert_results_entries.append([label, grouping, likert[label], likert_l[label], likert_h[label]])
            comparative_results_entries.append([label, grouping, comparative[label], comparative_l[label], comparative_h[label]])

likert_results_df = pd.DataFrame.from_records(likert_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
comparative_results_df = pd.DataFrame.from_records(comparative_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])

grouped_barplot_cohensd(likert_results_df, None, "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
                        filename='outputs/figures/likert_agreement_bar_full', plot_err=True,
                        fig_size=(20, 5), width=0.8)
grouped_barplot_cohensd(comparative_results_df, None, "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
                        filename='outputs/figures/comparative_agreement_bar_full', plot_err=True,
                        fig_size=(20, 5), width=0.8)

# Trimmed

likert_results_entries, comparative_results_entries = [], []
for grouping, agreements in result_dict.items():
    if not ('/' not in grouping and ('Dev' in grouping or 'Stu' in grouping)):
        likert = agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict()
        likert_h = agreements.xs('likert dialogue', level='category')["CI low"].to_dict()
        likert_l = agreements.xs('likert dialogue', level='category')["CI high"].to_dict()

        comparative = agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict()
        comparative_h = agreements.xs('comparative', level='category')["CI low"].to_dict()
        comparative_l = agreements.xs('comparative', level='category')["CI high"].to_dict()

        for label, score in likert.items():
            likert_results_entries.append([label, grouping, likert[label], likert_l[label], likert_h[label]])
            comparative_results_entries.append([label, grouping, comparative[label], comparative_l[label], comparative_h[label]])

likert_results_df = pd.DataFrame.from_records(likert_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
comparative_results_df = pd.DataFrame.from_records(comparative_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])

grouped_barplot_cohensd(likert_results_df, None, "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
                        filename='outputs/figures/likert_agreement_bar', plot_err=True,
                        fig_size=(20, 3), width=0.8)
grouped_barplot_cohensd(comparative_results_df, None, "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
                        filename='outputs/figures/comparative_agreement_bar', plot_err=True,
                        fig_size=(20, 3), width=0.8)


# Experts

# likert_results_entries, comparative_results_entries = [], []
# for grouping, agreements in result_dict.items():
#     if 'dev' in grouping.split('_'):
#         likert = agreements.xs('likert dialogue', level='category')["Krippendorff's alpha"].to_dict()
#         likert_h = agreements.xs('likert dialogue', level='category')["CI low"].to_dict()
#         likert_l = agreements.xs('likert dialogue', level='category')["CI high"].to_dict()
#
#         comparative = agreements.xs('comparative', level='category')["Krippendorff's alpha"].to_dict()
#         comparative_h = agreements.xs('comparative', level='category')["CI low"].to_dict()
#         comparative_l = agreements.xs('comparative', level='category')["CI high"].to_dict()
#
#         for label, score in likert.items():
#             likert_results_entries.append([label, grouping, likert[label], likert_l[label], likert_h[label]])
#             comparative_results_entries.append([label, grouping, comparative[label], comparative_l[label], comparative_h[label]])
#
# likert_results_df = pd.DataFrame.from_records(likert_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
# comparative_results_df = pd.DataFrame.from_records(comparative_results_entries, columns=["label", "groups", "estimate", "CI low", "CI high"])
#
# grouped_barplot_cohensd(likert_results_df, "l", "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
#                         filename='outputs/figures/likert_agreement_bar_expert', plot_err=True)
# grouped_barplot_cohensd(comparative_results_df, "c", "", "", ylim=(-0.4, 1.05), value_col="estimate", rot=0,
#                         filename='outputs/figures/comparative_agreement_bar_expert', plot_err=True)

#########################################
##  UI V SX - DIFFERENCE TEST
#########################################

# def choices_to_csv(df, type):
#     for cat in set(df.index.get_level_values(sym.category)):
#         for lab in set(df.index.get_level_values(sym.label)):
#             extracted = df.xs((cat, lab), level=(1, 2)).astype(int)
#             extracted.to_csv(f'outputs/csv/{cat}_{lab}_{type}.csv', index=False, header=False)
#
# def calc_proportion(df):
#     df['equal'] = (df[df.columns[0]] == df[df.columns[1]]).astype(int)
#     sum = df['equal'].sum()
#     total = df['equal'].shape[0]
#     return pd.DataFrame({f'{prop_type} sum': [sum], f'{prop_type} total': [total], f'prop': [sum/total]})
#
# prop_type = 'eve'
# external_and_external_double = get_doubly_annotated(sx_sx)
# # choices_to_csv(external_and_external_double, 'external_and_external')
# eve_label_groups = external_and_external_double.groupby(level=[sym.category, sym.label])
# external_and_external_proportions = eve_label_groups.apply(calc_proportion)
#
# prop_type = 'ive'
# # choices_to_csv(interactive_and_external, 'interactive_and_external')
# ive_label_groups = ui_sx_agreements.groupby(level=[sym.category, sym.label])
# interactive_and_external_proportions = ive_label_groups.apply(calc_proportion)
#
# all_proportions = pd.concat([interactive_and_external_proportions, external_and_external_proportions], axis=1)
#
# def run_ztests(df):
#     z, p = proportions_ztest(count=[df.loc['ive sum'], df.loc['eve sum']], nobs=[df.loc['ive total'], df.loc['eve total']])
#     return pd.Series([z,p], index=['statistic', 'pvalue'])
#
# ztests = all_proportions.apply(run_ztests, axis=1)
# ztests = ztests.droplevel(2)

#########################################
##  GRAPHING
#########################################

if PLOT:
    scatter_graph_with_error(ui_sx_agreements,
                             column="Krippendorff's alpha", ylabel=r"$\alpha$",
                             overlay=sx_sx_agreements,
                             # significance=ztests,
                             filename='outputs/figures/agreements_with_significance',
                             figsize=(12,4))

    # interactive_and_external_proportions['CI low'] = interactive_and_external_proportions['prop']
    # interactive_and_external_proportions['CI high'] = interactive_and_external_proportions['prop']
    # external_and_external_proportions['CI low'] = external_and_external_proportions['prop']
    # external_and_external_proportions['CI high'] = external_and_external_proportions['prop']
    #
    # scatter_graph_with_error(interactive_and_external_proportions,
    #                          column="prop", ylabel=r"Prop",
    #                          # overlay=external_and_external_proportions,
    #                          significance=ztests,
    #                          filename='outputs/figures/agreement_proportions_with_significance')
