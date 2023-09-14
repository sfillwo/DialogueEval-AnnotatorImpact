from utilities.analysis import *
from utilities.graphing import *
import pandas as pd


PLOT = False

#########################################
##  DATA
#########################################

interactive_annotations = data.dialogue_collection.annotation_dataframe()
interactive_annotations_comparative = data.dialogue_collection.comparative_annotation_dataframe()

external_annotations = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123)
external_annotations_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

# UX
ux = get_singly_annotated(data_student_extcomp.student_external_comparative.annotation_dataframe(), seed=123)
ux_comparative = get_singly_annotated(data_student_extcomp.student_external_comparative.comparative_annotation_dataframe(), seed=123)

#########################################
##  BOT-PAIR DIFFERENCES
#########################################

# table per metric
# annotator type vs bot-pair
# cell denotes 1 if statistically different, 0 otherwise

pvalues_external = p_values_comparing_bots(data.surge_evaluation).round(4)
pvalues_interactive = p_values_comparing_bots(data.dialogue_collection).round(4)
pvalues_student_ext = p_values_comparing_bots(data_student_extcomp.student_external_comparative).round(4).dropna(axis='columns')

def reorder(df, ordered_index):
    order = list(reversed(ordered_index))
    ordered_df = df[order]
    return ordered_df

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
sensitivities = {}
for idx in pvalues_external.index:
    ext_pvals = pvalues_external.loc[idx]
    int_pvals = pvalues_interactive.loc[idx]
    if idx in pvalues_student_ext.index:
        stu_ext_pvals = pvalues_student_ext.loc[idx]
    else: # dummy series for stu_ext_pvals for likert evaluations
        stu_ext_pvals = pd.Series([1.0 for k in ext_pvals.index], index=ext_pvals.index)
    combined = pd.concat([int_pvals, stu_ext_pvals, ext_pvals], axis=1)
    combined.columns = ['Ui', 'Ux', 'Sx']
    combined = combined.apply(lambda x: [1 if y < 0.05 else 0 for y in x])
    combined = combined.T
    combined['total'] = combined.apply(sum, axis=1)

    interactive_row = combined.loc['Ui']
    external_row = combined.loc['Sx']
    stu_external_row = combined.loc['Ux']
    # external_row_reordered = reorder(external_row, interactive_row.index)
    # stu_external_row_reordered = reorder(stu_external_row, interactive_row.index)
    # combined_row = pd.concat([interactive_row, stu_external_row_reordered, external_row_reordered])
    combined_row = pd.concat([interactive_row.loc['total'], stu_external_row.loc['total'], external_row.loc['total']])
    combined_row.index = ['Ui', 'Ux', 'Sx']
    subscript = idx[0].split(' ')[-1][0]
    tag = f"{dimensions_transformer[idx[1]]}$_{subscript}$"
    sensitivities[tag] = combined_row

sensitivities_df = pd.DataFrame(sensitivities).T
sensitivities_df = sensitivities_df.rename(columns=bot_tags)
sensitivities_df.to_csv('outputs/csv/sensitivities.csv')