from utilities.analysis import *
from utilities.graphing import *
from utilities.utils import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PLOT = True

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

#########################################
##  STANDARD DEVIATION OVER SIZE
#########################################

def get_std_by_size(annotations):
    size_to_std = {}
    size_to_mean = {}
    bots = set(annotations.index.get_level_values(0))
    for downsample in range(20,101,5):
        aggregate_stds = None
        aggregate_means = None
        for i in range(100):
            df = None
            for bot in bots:
                sampled = downsample_dials(annotations.xs(bot), downsample)
                if df is None:
                    df = sampled
                else:
                    df = pd.concat([df, sampled])
            label_groups = df.groupby(level=['category', 'label'])
            stds = label_groups.std()
            means = label_groups.mean()
            if aggregate_stds is None:
                aggregate_stds = stds
                aggregate_means = means
            else:
                aggregate_stds = pd.concat([aggregate_stds, stds], axis=1)
                aggregate_means = pd.concat([aggregate_means, means], axis=1)
        size_to_std[downsample] = aggregate_stds.mean(axis=1)
        size_to_mean[downsample] = aggregate_means.mean(axis=1)
    return size_to_std, size_to_mean

def get_std_by_size_by_bot(annotations):
    size_to_std = {}
    size_to_mean = {}
    bots = set(annotations.index.get_level_values(0))
    for downsample in range(20,101,5):
        for bot in bots:
            aggregate_stds = None
            aggregate_means = None
            for i in range(100):
                sampled = downsample_dials(annotations.xs(bot), downsample)
                label_groups = sampled.groupby(level=['category', 'label'])
                stds = label_groups.std()
                means = label_groups.mean()
                if aggregate_stds is None:
                    aggregate_stds = stds
                    aggregate_means = means
                else:
                    aggregate_stds = pd.concat([aggregate_stds, stds], axis=1)
                    aggregate_means = pd.concat([aggregate_means, means], axis=1)
            size_to_std.setdefault(bot, {})[downsample] = aggregate_stds.mean(axis=1)
            size_to_mean.setdefault(bot, {})[downsample] = aggregate_means.mean(axis=1)
    return size_to_std, size_to_mean

external_std_to_size_to_bot, external_mean_to_size_to_bot = get_std_by_size_by_bot(external_annotations)
interactive_std_to_size_to_bot, interactive_mean_to_size_to_bot = get_std_by_size_by_bot(interactive_annotations)

external_std_to_size, external_mean_to_size = get_std_by_size(external_annotations)
interactive_std_to_size, interactive_mean_to_size = get_std_by_size(interactive_annotations)


def plot_curves(interactive_size_to_sensitivity, external_size_to_sensitivity, filename):
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
    def plot_size_to_sensitivity(size_to_sensitivity, type):
        xs = [] # dialogue #
        ys = {} # dimension: sensitivity_proportion
        for size, df in sorted(size_to_sensitivity.items()):
            xs.append(size)
            for idx in df.index:
                dim = idx[1]
                ys.setdefault(dim, []).append(df.loc[idx])
        for dim, y in ys.items():
            ax.plot(xs, y, color=graphing_dim_colors[dim], linestyle=linestyles[type], linewidth=3)
        return ys.keys()
    plot_size_to_sensitivity(interactive_size_to_sensitivity, 'interactive')
    dimensions = plot_size_to_sensitivity(external_size_to_sensitivity, 'external')

    custom_lines = [
        Line2D([0], [0], linestyle=linestyles['interactive'], color='black', lw=2),
        Line2D([0], [0], linestyle=linestyles['external'], color='black', lw=2),
        *[Patch(facecolor=graphing_dim_colors[dim]) for dim in dimensions]
    ]

    tagged_dims = []
    for dim in dimensions:
        tagged_dims.append(dimensions_transformer[dim])

    ax.legend(custom_lines,
              ['Interactive', 'External', *tagged_dims],
              handlelength=2, ncol=5, loc='upper left',
              bbox_to_anchor=(0, 1.12))

    ax.set_xlabel('Dialogues', labelpad=20)
    # ax.set_ylabel('Std', rotation=90, labelpad=20)

    if PLOT:
        plt.savefig(filename, format="png", bbox_inches="tight")

    plt.show()

if PLOT:
    for bot in interactive_std_to_size_to_bot:
        print(bot)
        plot_curves(interactive_std_to_size_to_bot[bot], external_std_to_size_to_bot[bot], filename=f"outputs/figures/std_analysis_{bot}.png")
        plot_curves(interactive_mean_to_size_to_bot[bot], external_mean_to_size_to_bot[bot], filename=f"outputs/figures/mean_analysis_{bot}.png")

    plot_curves(interactive_std_to_size, external_std_to_size, filename=f"outputs/figures/std_analysis.png")
    plot_curves(interactive_mean_to_size, external_mean_to_size, filename=f"outputs/figures/mean_analysis.png")