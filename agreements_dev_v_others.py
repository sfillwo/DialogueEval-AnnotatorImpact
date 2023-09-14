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
##  Dev V Others AGREEMENTS
#########################################

dev_ui = pd.concat([developer, interactive_annotations], axis=1).dropna()
dev_ui.columns = ['dev', 'other']

dev_sx = pd.concat([developer, external_annotations], axis=1).dropna()
dev_sx.columns = ['dev', 'other']

dev_ndev = pd.concat([developer, nondeveloper], axis=1).dropna()
dev_ndev.columns = ['dev', 'other']

dev_others = pd.concat([dev_ui, dev_sx, dev_ndev], axis=0)

dev_others_agreements = agreement_dataframe(
    dev_others, load='outputs/results/dev_others_krippendorf'
)

#########################################
##  Others V Others AGREEMENTS
#########################################

ndev_ui = pd.concat([nondeveloper, interactive_annotations], axis=1).dropna()
ndev_ui.columns = ['one', 'two']

ndev_sx = pd.concat([nondeveloper, external_annotations], axis=1).dropna()
ndev_sx.columns = ['one', 'two']

ui_sx = extract_just_likert_comparative(external_annotations, interactive_annotations)
ui_sx.columns = ['one', 'two']

other_others = pd.concat([ndev_ui, ndev_sx, ui_sx], axis=0)

other_other_agreements = agreement_dataframe(
    other_others, load='outputs/results/other_others_krippendorf'
)



#########################################
##  Table
#########################################

result_dict = {
    "Expert-Novice": dev_others_agreements,
    "Novice": other_other_agreements
}

likert_results_entries, comparative_results_entries = [], []
for grouping, agreements in result_dict.items():
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

grouped_barplot_cohensd(likert_results_df, "l", "", "", ylim=(-0.1, 0.5), value_col="estimate", rot=0,
                        filename='outputs/figures/likert_agreement_bar_experts', plot_err=True)
grouped_barplot_cohensd(comparative_results_df, "c", "", "", ylim=(-0.1, 0.5), value_col="estimate", rot=0,
                        filename='outputs/figures/comparative_agreement_bar_experts', plot_err=True)