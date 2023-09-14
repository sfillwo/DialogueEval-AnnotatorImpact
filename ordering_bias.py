import pandas as pd

from utilities.analysis import *
from utilities.graphing import *
from collections import defaultdict
from scipy.stats import ttest_ind

PLOT = False

#########################################
##  DATA
#########################################

external_annotations = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123).drop(
    index=category.behavior, level=sym.category, errors='ignore'
).drop(
    index=category.comparative, level=sym.category
).drop(
    index=category.likert_turn, level=sym.category, errors='ignore'
)
external_annotations_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

interactive_annotations = data.dialogue_collection.annotation_dataframe().drop(
    index=category.comparative, level=sym.category
)
interactive_annotations_comparative = data.dialogue_collection.comparative_annotation_dataframe()

ux_annotations = get_singly_annotated(data_student_extcomp.student_external_comparative.annotation_dataframe(), seed=123)
ux_annotations_comparative = get_singly_annotated(data_student_extcomp.student_external_comparative.comparative_annotation_dataframe(), seed=123)

#########################################
##  INTERACTIVE LIKERT ORDER EFFECTS
#########################################

def get_bot(id):
    if 'emora' in id:
        return 'Emora'
    elif 'blender2' in id:
        return 'Blender2'
    elif 'bart' in id:
        return 'BART-FiD-RAG'
    elif 'rerank' in id:
        return 'Blender-Decode'

first_conversations = {unit.dialogue_ids[0] for _, unit in data.dialogue_collection.work_units.items()}
first_counts = defaultdict(int)
for id in first_conversations:
    first_counts[get_bot(id)] += 1

second_conversations = {unit.dialogue_ids[1] for _, unit in data.dialogue_collection.work_units.items()}
second_counts = defaultdict(int)
for id in second_conversations:
    second_counts[get_bot(id)] += 1

pairs_by_first = {unit.dialogue_ids[0]: unit.dialogue_ids[1] for _, unit in data.dialogue_collection.work_units.items()}
pairs_by_second = {v: k for k, v in pairs_by_first.items()}

first_likert_annotations = interactive_annotations[interactive_annotations.index.get_level_values("item").isin(first_conversations)]
second_likert_annotations = interactive_annotations[interactive_annotations.index.get_level_values("item").isin(second_conversations)]

def likert_order_effects(first_likert_annotations, second_likert_annotations):
    first_ratings = aggregate_likert_ratings(
        first_likert_annotations, category.likert_dialogue,
        reload='outputs/results/interactive_likert_dialogue_ratings_orderedfirst'
    )

    second_ratings = aggregate_likert_ratings(
        second_likert_annotations, category.likert_dialogue,
        reload='outputs/results/interactive_likert_dialogue_ratings_orderedsecond'
    )

    first_ratings_overall = aggregate_likert_ratings(
        first_likert_annotations, category.likert_dialogue,
        reload='outputs/results/interactive_likert_dialogue_ratings_orderedfirst_overall',
        bot_groups=False
    )
    first_ratings_overall_records = [('first', *rec) for rec in first_ratings_overall.to_records()]

    second_ratings_overall = aggregate_likert_ratings(
        second_likert_annotations, category.likert_dialogue,
        reload='outputs/results/interactive_likert_dialogue_ratings_orderedsecond_overall',
        bot_groups=False
    )
    second_ratings_overall_records = [('second', *rec) for rec in second_ratings_overall.to_records()]

    overall_likert_by_order = pd.DataFrame.from_records(first_ratings_overall_records + second_ratings_overall_records,
                                                        columns=['bot', 'label', 'mean', 'CI low', 'CI high', 'n'])
    overall_likert_by_order.set_index(['bot', 'label'], inplace=True)

    for label in set(first_likert_annotations.index.get_level_values('label')):
        a = first_likert_annotations.xs(label, level='label')
        b = second_likert_annotations.xs(label, level='label')
        results = ttest_ind(a, b, equal_var=False)
        print(label, f"({first_ratings_overall.loc[label]['mean']:.2f}, {second_ratings_overall.loc[label]['mean']:.2f})", f"p={results[1][0]:.3f}")

    if PLOT:
        grouped_barplot(first_ratings, subscript='d', ylabel=None, xlabel=None, ylim=(2.5, 5), rot=0, fig_size=(10,3), filename='outputs/figures/interactive_likert_dialogue_orderedfirst')
        grouped_barplot(second_ratings, subscript='d', ylabel=None, xlabel=None, ylim=(2.5, 5), rot=0, fig_size=(10,3), filename='outputs/figures/interactive_likert_dialogue_orderedsecond')
        grouped_barplot(overall_likert_by_order, subscript='d', ylabel=None, xlabel=None, ylim=(2.5, 5), rot=0, fig_size=(10,3), filename='outputs/figures/interactive_likert_dialogue_order_effect')

print('UI Likert')
likert_order_effects(first_likert_annotations, second_likert_annotations)

#########################################
##  INTERACTIVE COMPARATIVE ORDER EFFECTS
#########################################

first_comparative_annotations = interactive_annotations_comparative[interactive_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_first.items()])]
second_comparative_annotations = interactive_annotations_comparative[interactive_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_second.items()])]

def comparative_order_effects(first_comparative_annotations, second_comparative_annotations):
    first_comparative = aggregate_comparative(
        first_comparative_annotations,
        reload='outputs/results/interactive_comparative_orderedfirst'
    )
    botvothers = first_comparative[first_comparative.index.get_level_values('bot comp') == 'others'][['win', 'tie', 'lose']]
    botvothers['CI low'] = first_comparative.iloc[:, 9]
    botvothers['CI high'] = first_comparative.iloc[:, 10]
    botvothers.reset_index(level=['bot comp'], inplace=True)
    botvothers.drop('bot comp', inplace=True, axis='columns')
    first_toplot = botvothers.reorder_levels(['label', 'bot']).reindex(["BART-FiD-RAG", 'Blender2', 'Emora', 'Blender-Decode'], level=1)

    second_comparative = aggregate_comparative(
        second_comparative_annotations,
        reload='outputs/results/interactive_comparative_orderedsecond'
    )
    botvothers = second_comparative[second_comparative.index.get_level_values('bot comp') == 'others'][['win', 'tie', 'lose']]
    botvothers['CI low'] = second_comparative.iloc[:, 9]
    botvothers['CI high'] = second_comparative.iloc[:, 10]
    botvothers.reset_index(level=['bot comp'], inplace=True)
    botvothers.drop('bot comp', inplace=True, axis='columns')
    second_toplot = botvothers.reorder_levels(['label', 'bot']).reindex(["BART-FiD-RAG", 'Blender2', 'Emora', 'Blender-Decode'], level=1)

    if PLOT:
        plot_comparative(first_toplot, 'c', 'Comparative Evaluation Results', 'win', (10, 5), filename='outputs/figures/interactive_comparative_orderedfirst')
        plot_comparative(second_toplot, 'c', 'Comparative Evaluation Results', 'win', (10, 5), filename='outputs/figures/interactive_comparative_orderedsecond')

    for label in set(first_comparative_annotations.index.get_level_values('label')):
        x = first_comparative_annotations.xs(label, level='label')
        x_noties = x[x != 0]
        print()
        print(label)
        print(x_noties.value_counts(normalize=True))

print('\nUI Comparative')
comparative_order_effects(first_comparative_annotations, second_comparative_annotations)

#########################################
##  EXTERNAL COMPARATIVE ORDER EFFECTS
#########################################

pairs_by_first = {unit.dialogue_ids[0]: unit.dialogue_ids[1] for _, unit in data.surge_evaluation.work_units.items() if unit.task == 'comparative'}
pairs_by_second = {v: k for k, v in pairs_by_first.items()}

first_comparative_annotations = external_annotations_comparative[external_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_first.items()])]
second_comparative_annotations = external_annotations_comparative[external_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_second.items()])]

print('\nSX Comparative')
comparative_order_effects(first_comparative_annotations, second_comparative_annotations)

pairs_by_first = {unit.dialogue_ids[0]: unit.dialogue_ids[1] for _, unit in data_student_extcomp.student_external_comparative.work_units.items() if unit.task == 'comparative'}
pairs_by_second = {v: k for k, v in pairs_by_first.items()}

first_comparative_annotations = ux_annotations_comparative[ux_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_first.items()])]
second_comparative_annotations = ux_annotations_comparative[ux_annotations_comparative.index.get_level_values("dialogues").isin([(k,v) for k,v in pairs_by_second.items()])]

print('\nUX Comparative')
comparative_order_effects(first_comparative_annotations, second_comparative_annotations)