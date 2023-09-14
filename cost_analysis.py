import matplotlib.pyplot as plt

from utilities.analysis import *
from utilities.graphing import *
import pandas as pd
import json

PLOT = True

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
dx = get_singly_annotated(developer_ext.student_external.annotation_dataframe(), seed=123)
dx_comparative = get_singly_annotated(developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

# Non developers
ux = get_singly_annotated(non_developer_ext.student_external.annotation_dataframe(), seed=123)
ux_comparative = get_singly_annotated(non_developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

#########################################
##  BOT-PAIR DIFFERENCES - COST COMPARISON
#########################################
import os, time
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_sensitivity_curves(interactive_size_to_sensitivity, external_size_to_sensitivity, column, filename):
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=28)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["figure.figsize"] = (10, 10)

    linestyles = {'external': 'dotted',
                  'interactive': 'solid'}
    fig, ax = plt.subplots()
    def plot_size_to_sensitivity(size_to_sensitivity, column, type):
        xs = [] # dialogue #
        ys = {} # botpair: sensitivity_proportion
        for size, sensitivity_df in sorted(size_to_sensitivity.items()):
            xs.append(size)
            for botpair in sensitivity_df.columns:
                ordered_botpair = tuple(sorted(botpair))
                ys.setdefault(ordered_botpair, []).append(sensitivity_df.loc[('likert dialogue', column)][botpair] / 1000)
        for botpair, y in ys.items():
            ax.plot(xs, y, color=graphing_botpair_colors[botpair], linestyle=linestyles[type], linewidth=3)
        return ys.keys()
    plot_size_to_sensitivity(interactive_size_to_sensitivity, column, 'interactive')
    botpairs = plot_size_to_sensitivity(external_size_to_sensitivity, column, 'external')


    custom_lines = [
        Line2D([0], [0], linestyle=linestyles['interactive'], color='black', lw=2),
        Line2D([0], [0], linestyle=linestyles['external'], color='black', lw=2),
        *[Patch(facecolor=graphing_botpair_colors[botpair]) for botpair in botpairs]
    ]

    tagged_botpairs = []
    for bp in botpairs:
        b1, b2 = bp
        tagged_botpairs.append(f"{bot_tags[b1]}/{bot_tags[b2]}")

    ax.legend(custom_lines,
              ['Interactive', 'External', *tagged_botpairs],
              handlelength=2, ncol=4, loc='upper left',
              bbox_to_anchor=(0, 1.12))

    ax.set_xlabel('Dialogues', labelpad=20)
    ax.set_ylabel('Sensitivity', rotation=90, labelpad=20)

    if PLOT:
        plt.savefig(filename, format="png", bbox_inches="tight")

    plt.tight_layout()
    plt.show()

def likert_sensitivity_costs_botpair(evaluation, type, load=False):
    # preprocess df once
    annotations = get_singly_annotated(evaluation.annotation_dataframe(), seed=123)
    mean_annotations = annotations.drop(
        index=category.behavior, level=sym.category, errors='ignore'
    ).drop(
        index=category.comparative, level=sym.category
    ).drop(
        index=category.likert_turn, level=sym.category, errors='ignore'
    )

    size_to_sensitivity = {}
    dir = f"outputs/csv/cost_analysis_sensitivity/{type}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if not load:
        for size in range(105, 201, 5):
            print(size, end=' ')
            start = time.time()
            summed_sensitivity_counts = None
            for i in range(1000):
                sampled_pvalues = p_values_comparing_bots(evaluation, exclude_comparative=True, downsample=size, replace=True,
                                                          deterministic=False, mean_annotations=mean_annotations)
                sampled_sensitivity_counts = sampled_pvalues.apply(lambda x: [1 if y < 0.05 else 0 for y in x])
                if summed_sensitivity_counts is None:
                    summed_sensitivity_counts = sampled_sensitivity_counts
                else:
                    summed_sensitivity_counts += sampled_sensitivity_counts
            size_to_sensitivity[size] = summed_sensitivity_counts
            summed_sensitivity_counts.to_csv(f'{dir}/{size}_sensitivity.csv')
            print(f"- Elapsed: {(time.time() - start):.2f} s")
    else:
        for file in os.listdir(dir):
            size = int(file.split("_")[0])
            size_to_sensitivity[size] = pd.read_csv(f"{dir}/{file}", index_col=[0,1], header=[0,1])
    return size_to_sensitivity


# external_size_to_sensitivity = likert_sensitivity_costs_botpair(data.surge_evaluation, 'external_with_replace', load=True)
# interactive_size_to_sensitivity = likert_sensitivity_costs_botpair(data.dialogue_collection, 'interactive_with_replace', load=True)
#
# if PLOT:
#     plot_sensitivity_curves(interactive_size_to_sensitivity, external_size_to_sensitivity, 'quality',
#                             filename='outputs/figures/sensitivity_cost_analysis_with_replace_2.png')

#####################################################################################################################

def plot_sensitivity_curves_from_dict(filename, interactive_size_to_sensitivity, external_size_to_sensitivity, ux_size_to_sensitivity, dx_size_to_sensitivity):
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=28)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["figure.figsize"] = (10, 10)

    linestyles = {'Sx': 'dotted',
                  'Ui': 'solid',
                  'Ux': 'dashed',
                  'Dx': 'dashdot'}
    fig, ax = plt.subplots()
    def plot_size_to_sensitivity(size_to_sensitivity, type):
        xs = [] # dialogue #
        ys = []
        for size, sensitivity in sorted([(int(x[0]),x[1]) for x in size_to_sensitivity.items()]):
            xs.append(size)
            ys.append(sensitivity)
        ax.plot(xs, ys, linestyle=linestyles[type], linewidth=3)
    plot_size_to_sensitivity(interactive_size_to_sensitivity, 'Ui')
    plot_size_to_sensitivity(external_size_to_sensitivity, 'Sx')
    plot_size_to_sensitivity(ux_size_to_sensitivity, 'Ux')
    plot_size_to_sensitivity(dx_size_to_sensitivity, 'Dx')

    custom_lines = [
        Line2D([0], [0], linestyle=linestyles['Ui'], color='black', lw=2),
        Line2D([0], [0], linestyle=linestyles['Sx'], color='black', lw=2),
        Line2D([0], [0], linestyle=linestyles['Ux'], color='black', lw=2),
        Line2D([0], [0], linestyle=linestyles['Dx'], color='black', lw=2),
    ]

    legend_labels = ['Ui', 'Sx', 'Ux', 'Dx']
    ax.legend(custom_lines, legend_labels)

    ax.set_xlabel('Dialogues', labelpad=20)
    ax.set_ylabel('Sensitivity', rotation=90, labelpad=20)

    if PLOT:
        plt.savefig(filename, format="png", bbox_inches="tight")

    plt.tight_layout()
    plt.show()

def sensitivity_costs(evaluation, label, type, load=False):
    # preprocess df once
    annotations = get_singly_annotated(evaluation.annotation_dataframe(), seed=123)
    mean_annotations = annotations.drop(
        index=category.behavior, level=sym.category, errors='ignore'
    ).drop(
        index=category.comparative, level=sym.category
    ).drop(
        index=category.likert_turn, level=sym.category, errors='ignore'
    )
    mean_annotations = mean_annotations.xs(('likert dialogue', label), level=('category', 'label'), drop_level=False)

    size_to_sensitivity = {}
    dir = f"outputs/csv/cost_analysis_sensitivity/{label}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    dir = f"{dir}/{type}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if not load:
        for size in range(10, 201, 5):
            print(size, end=' ')
            start = time.time()
            summed_sensitivity_counts = []
            for i in range(1000):
                sampled_pvalues = p_values_comparing_bots(evaluation, downsample=size, replace=True,
                                                          deterministic=False, mean_annotations=mean_annotations,
                                                          exclude_comparative=True)
                sampled_sensitivity_counts = sampled_pvalues.apply(lambda x: [1 if y < 0.05 else 0 for y in x])
                summed_sensitivity_counts += [sampled_sensitivity_counts.mean(axis=1)[('likert dialogue', label)]]
            size_to_sensitivity[size] = np.mean(summed_sensitivity_counts)
            print(f"- Elapsed: {(time.time() - start):.2f} s")
        json.dump(size_to_sensitivity, open(f'{dir}/sensitivity.json', 'w'), indent=2)
    else:
        with open(f'{dir}/sensitivity.json') as f:
            size_to_sensitivity = json.load(f)
    return size_to_sensitivity

for label in ['quality', 'grammatical', 'consistent', 'informative', 'engaging', 'relevant', 'proactive', 'emotional']:
    sx_size_to_sensitivity = sensitivity_costs(data.surge_evaluation, type='external_with_replace_perc', load=True, label=label)
    ui_size_to_sensitivity = sensitivity_costs(data.dialogue_collection, type='interactive_with_replace_perc', load=True, label=label)
    ux_size_to_sensitivity = sensitivity_costs(non_developer_ext.student_external, type='ux_with_replace_perc', load=True, label=label)
    dx_size_to_sensitivity = sensitivity_costs(developer_ext.student_external, type='dx_with_replace_perc', load=True, label=label)

    if PLOT:
        plot_sensitivity_curves_from_dict(f'outputs/figures/sensitivity_cost_analysis_with_replace_perc_{label}.png',
                                          ui_size_to_sensitivity, sx_size_to_sensitivity,
                                          ux_size_to_sensitivity, dx_size_to_sensitivity)

#### COMP ###

def comparative_sensitivity_costs(evaluation, type, load=False):
    comp_annotations = get_singly_annotated(evaluation.comparative_annotation_dataframe(), seed=123)
    comp_annotations = comp_annotations.xs(('quality'), level=('label'), drop_level=False)

    size_to_sensitivity = {}
    dir = f"outputs/csv/cost_analysis_sensitivity/{type}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if not load:
        for size in range(10, 151, 5):
            print(size, end=' ')
            start = time.time()
            summed_sensitivity_counts = []
            for i in range(1000):
                sampled_pvalues = p_values_comparing_bots_comparative(evaluation, downsample=size, replace=True,
                                                                      deterministic=False, comp_annotations=comp_annotations)
                sampled_sensitivity_counts = sampled_pvalues.apply(lambda x: [1 if y < 0.05 else 0 for y in x])
                summed_sensitivity_counts += [sampled_sensitivity_counts.mean(axis=1)[('comparative', 'quality')]]
            size_to_sensitivity[size] = np.mean(summed_sensitivity_counts)
            print(f"- Elapsed: {(time.time() - start):.2f} s")
        json.dump(size_to_sensitivity, open(f'{dir}/sensitivity.json', 'w'), indent=2)
    else:
        with open(f'{dir}/sensitivity.json') as f:
            size_to_sensitivity = json.load(f)
    return size_to_sensitivity

external_size_to_sensitivity = comparative_sensitivity_costs(data.surge_evaluation, 'external_with_replace_comp_perc',
                                                 load=True)
interactive_size_to_sensitivity = comparative_sensitivity_costs(data.dialogue_collection, 'interactive_with_replace_comp_perc',
                                                    load=True)
external_student_size_to_sensitivity = comparative_sensitivity_costs(data_student_extcomp.student_external_comparative,
                                                         'extstudent_with_replace_comp_perc', load=True)

if PLOT:
    plot_sensitivity_curves_from_dict('outputs/figures/sensitivity_cost_analysis_with_replace_comp_perc.png',
                                      interactive_size_to_sensitivity, external_size_to_sensitivity,
                                      external_student_size_to_sensitivity)