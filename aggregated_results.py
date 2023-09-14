from utilities.analysis import *
from utilities.graphing import *
from utilities.utils import *
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.proportion import proportions_ztest

PLOT = True

#########################################
##  DATA
#########################################

interactive_annotations = data.dialogue_collection.annotation_dataframe()
interactive_annotations_comparative = data.dialogue_collection.comparative_annotation_dataframe()

external_annotations = get_singly_annotated(data.surge_evaluation.annotation_dataframe(), seed=123)
external_annotations_comparative = get_singly_annotated(data.surge_evaluation.comparative_annotation_dataframe(), seed=123)

# Developers
developer = get_singly_annotated(developer_ext.student_external.annotation_dataframe(), seed=123)
developer_comparative = get_singly_annotated(developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

# Non developers
nondeveloper = get_singly_annotated(non_developer_ext.student_external.annotation_dataframe(), seed=123)
nondeveloper_comparative = get_singly_annotated(non_developer_ext.student_external.comparative_annotation_dataframe(), seed=123)

#########################################
##  LIKERT DIALOGUE
#########################################

interactive_likert_dialogue_ratings = aggregate_likert_ratings(
    interactive_annotations, category.likert_dialogue,
    load='outputs/results/interactive_likert_dialogue_ratings'
)
interactive_likert_dialogue_ratings = interactive_likert_dialogue_ratings.drop('n', axis=1)
# interactive_likert_dialogue_ratings.columns=['int mean', 'int low', 'int high']

external_likert_dialogue_ratings = aggregate_likert_ratings(
    external_annotations, category.likert_dialogue,
    load='outputs/results/external_likert_dialogue_ratings'
)
external_likert_dialogue_ratings = external_likert_dialogue_ratings.drop('n', axis=1)
# external_likert_dialogue_ratings.columns=['ext mean', 'ext low', 'ext high']

developer_likert_dialogue_ratings = aggregate_likert_ratings(
    developer, category.likert_dialogue,
    reload='outputs/results/developer_likert_dialogue_ratings'
)
developer_likert_dialogue_ratings = developer_likert_dialogue_ratings.drop('n', axis=1)

nondeveloper_likert_dialogue_ratings = aggregate_likert_ratings(
    nondeveloper, category.likert_dialogue,
    reload='outputs/results/nondeveloper_likert_dialogue_ratings'
)
nondeveloper_likert_dialogue_ratings = nondeveloper_likert_dialogue_ratings.drop('n', axis=1)

if PLOT:
    grouped_barplot(interactive_likert_dialogue_ratings, subscript=None, ylabel=None, xlabel=None, ylim=(2.0, 5), rot=0, fig_size=(10,3), filename='outputs/figures/interactive_likert_dialogue')
    grouped_barplot(external_likert_dialogue_ratings, subscript=None, ylabel=None, xlabel=None, ylim=(2.0, 5), rot=0, fig_size=(10,3), filename='outputs/figures/external_likert_dialogue')
    grouped_barplot(developer_likert_dialogue_ratings, subscript=None, ylabel=None, xlabel=None, ylim=(2.0, 5), rot=0, fig_size=(10, 3), filename='outputs/figures/developer_likert_dialogue')
    grouped_barplot(nondeveloper_likert_dialogue_ratings, subscript=None, ylabel=None, xlabel=None, ylim=(2.0, 5), rot=0, fig_size=(10, 3), filename='outputs/figures/nondeveloper_likert_dialogue')

#########################################
##  LIKERT DIALOGUE - INTERACTIVE-VS-EXTERNAL BY-BOT
#########################################

def paired_ttest(df):
    s, p = ttest_rel(df['external'], df['interactive'])
    return pd.Series([s, p], index=['statistic', 'pvalue'])

interactive_and_external = extract_just_likert_comparative(external_annotations, interactive_annotations)
interactive_groups = interactive_and_external.groupby(level=[sym.category, sym.label, sym.bot])
paired_ttests = interactive_groups.apply(paired_ttest)

def aggregate_by_bot(interactive_df, external_df):
    bots = set(interactive_df.index.get_level_values('bot'))
    for bot in bots:
        print('Likert Dialogue Rating: ', bot)
        idf = interactive_df.xs(bot, level='bot')
        edf = external_df.xs(bot, level='bot')
        idf_extended = pd.concat({'likert dialogue': idf}, names=['category'])
        edf_extended = pd.concat({'likert dialogue': edf}, names=['category'])
        significance = paired_ttests.xs(bot, level='bot')
        scatter_graph_with_error(idf_extended, column=stat.mean,
                                 ylabel=None, overlay=edf_extended,
                                 ylim=(2.5, 5.0), figsize=(10,5),
                                 xaxis_colored=False, filename=f"outputs/figures/{bot}_likert_dialogue",
                                 significance=significance, single_color='red')
if False:
    aggregate_by_bot(interactive_likert_dialogue_ratings, external_likert_dialogue_ratings)

#########################################
##  COMPARATIVE
#########################################

def get_comparative_to_plot(df):
    botvothers = df[df.index.get_level_values('bot comp') == 'others'][['win', 'tie', 'lose']]
    botvothers['CI low'] = df.iloc[:, 9]
    botvothers['CI high'] = df.iloc[:, 10]
    botvothers.reset_index(level=['bot comp'], inplace=True)
    botvothers.drop('bot comp', inplace=True, axis='columns')
    to_plot = botvothers.reorder_levels(['label', 'bot']).reindex(["BART-FiD-RAG", 'Blender2', 'Emora', 'Blender-Decode'], level=1)
    return to_plot

interactive_comparative_df = aggregate_comparative(
    interactive_annotations_comparative,
    load='outputs/results/interactive_comparative'
)
interactive_toplot = get_comparative_to_plot(interactive_comparative_df)

external_comparative_df = aggregate_comparative(
    external_annotations_comparative,
    load='outputs/results/external_comparative'
)
external_toplot = get_comparative_to_plot(external_comparative_df)

developer_comparative_df = aggregate_comparative(
    developer_comparative,
    reload='outputs/results/developer_comparative'
)
developer_toplot = get_comparative_to_plot(developer_comparative_df)

nondeveloper_comparative_df = aggregate_comparative(
    nondeveloper_comparative,
    reload='outputs/results/nondeveloper_comparative'
)
nondeveloper_toplot = get_comparative_to_plot(nondeveloper_comparative_df)


if True:
    plot_comparative(interactive_toplot, None, 'Comparative Evaluation Results', 'win', (10, 3), ylim=(0, 0.9), filename='outputs/figures/interactive_comparative')
    plot_comparative(external_toplot, None, 'Comparative Evaluation Results', 'win', (10, 3), ylim=(0, 0.9), filename='outputs/figures/external_comparative')
    plot_comparative(developer_toplot, None, 'Comparative Evaluation Results', 'win', (10, 3), ylim=(0, 0.9), filename='outputs/figures/developer_comparative')
    plot_comparative(nondeveloper_toplot, None, 'Comparative Evaluation Results', 'win', (10, 3), ylim=(0, 0.9), filename='outputs/figures/nondeveloper_comparative')

#########################################
##  COMPARATIVE - INTERACTIVE-VS-EXTERNAL BY-BOT
#########################################

def run_ztests(df):
    # print(set(df.index.get_level_values('bot')), set(df.index.get_level_values('label')))
    int_win_prop = df['interactive'][df['interactive'] == 1].shape[0]
    int_tie_prop = df['interactive'][df['interactive'] == 0].shape[0]
    int_lose_prop = df['interactive'][df['interactive'] == -1].shape[0]
    ext_win_prop = df['external'][df['external'] == 1].shape[0]
    ext_tie_prop = df['external'][df['external'] == 0].shape[0]
    ext_lose_prop = df['external'][df['external'] == -1].shape[0]
    win_z, win_p = proportions_ztest(count=[int_win_prop, ext_win_prop], nobs=[df.shape[0], df.shape[0]])
    tie_z, tie_p = proportions_ztest(count=[int_tie_prop, ext_tie_prop], nobs=[df.shape[0], df.shape[0]])
    lose_z, lose_p = proportions_ztest(count=[int_lose_prop, ext_lose_prop], nobs=[df.shape[0], df.shape[0]])
    return pd.Series([win_z, win_p, tie_z, tie_p, lose_z, lose_p], index=['win statistic', 'win pvalue', 'tie statistic', 'tie pvalue', 'lose statistic', 'lose pvalue'])

interactive_and_external = extract_just_likert_comparative(external_annotations, interactive_annotations).xs('comparative', level='category')
interactive_groups = interactive_and_external.groupby(level=[sym.label, sym.bot])
ztest_results = interactive_groups.apply(run_ztests)

def ztest_plot_by_bot(interactive_df, external_df):
    ci_idx = {
        'win': (9, 10),
        'tie': (5, 6),
        'lose': (1, 2)
    }
    fmt_map = {
        'win': 'o',
        'tie': 's',
        'lose': 'v'
    }
    bots = set(interactive_df.index.get_level_values('bot'))
    for bot in bots:
        print('Comparative: ', bot)
        plt.rcParams["figure.figsize"] = (10,4)
        fig, axs = plt.subplots()
        axs = [axs]
        for i, decision in enumerate(['win']): #, 'tie', 'lose']):
            int_botvothers = interactive_df[interactive_df.index.get_level_values('bot comp') == 'others'][['win', 'tie', 'lose']]
            int_botvothers['CI low'] = interactive_df.iloc[:, ci_idx[decision][0]]
            int_botvothers['CI high'] = interactive_df.iloc[:, ci_idx[decision][1]]
            int_botvothers.reset_index(level=['bot comp'], inplace=True)
            int_botvothers.drop('bot comp', inplace=True, axis='columns')
            interactive_toplot = int_botvothers.reorder_levels(['label', 'bot']).reindex(["BART-FiD-RAG", 'Blender2', 'Emora', 'Blender-Decode'], level=1)

            ext_botvothers = external_df[external_df.index.get_level_values('bot comp') == 'others'][['win', 'tie', 'lose']]
            ext_botvothers['CI low'] = external_df.iloc[:, ci_idx[decision][0]]
            ext_botvothers['CI high'] = external_df.iloc[:, ci_idx[decision][1]]
            ext_botvothers.reset_index(level=['bot comp'], inplace=True)
            ext_botvothers.drop('bot comp', inplace=True, axis='columns')
            external_toplot = ext_botvothers.reorder_levels(['label', 'bot']).reindex(["BART-FiD-RAG", 'Blender2', 'Emora', 'Blender-Decode'], level=1)

            idf = interactive_toplot.xs(bot, level='bot')[[decision, "CI low", "CI high"]]
            edf = external_toplot.xs(bot, level='bot')[[decision, "CI low", "CI high"]]
            idf_extended = pd.concat({'comparative': idf}, names=['category'])
            edf_extended = pd.concat({'comparative': edf}, names=['category'])
            significance = ztest_results.xs(bot, level='bot')[[f'{decision} statistic', f'{decision} pvalue']]
            significance.columns = ['statistic', 'pvalue']
            significance = pd.concat({'comparative': significance}, names=['category'])
            scatter_graph_with_error(idf_extended, column=decision,
                                     ylabel=None, overlay=edf_extended,
                                     rot=0, ylim=(0, 0.8),
                                     xaxis_colored=False, filename=f"outputs/figures/{bot}_comparative_{decision}",
                                     significance=significance, single_color='red', fmt=fmt_map[decision], ax=axs[i])
        plt.savefig(f"outputs/figures/{bot}_comparative" + '.png', format="png", bbox_inches="tight")
        plt.show()

if False:
    ztest_plot_by_bot(interactive_comparative_df, external_comparative_df)