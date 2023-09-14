
import nltk
import random
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import bootstrap as retarded_bootstrap
from scipy.stats import ttest_ind, spearmanr, pearsonr
import krippendorff
import utilities.evaluation_data_definitions as edd
from attrs import define
from cattrs import structure
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import binom_test
from itertools import combinations
from utilities.utils import *

from utilities.evaluation_data_definitions import Project, Evaluation
import pathlib
proj_dir = pathlib.Path(__file__).parent.parent.resolve()

@define
class StudentExtProject:
    student_external: Evaluation | None = None

with open(f'{proj_dir}/data/data.json') as f:
    json_data = json.load(f)
    data = structure(json_data, Project)

    # exclude from interactive and external students those tasks missing from external surge
    surge = data.surge_evaluation.annotation_dataframe()
    comparative_surge_dialogue_ids = {item[1] for item in surge.xs(('comparative', 'informative'),
                                                  level=['category', 'label']).index.tolist()}
    likert_surge_dialogue_ids = {item[1] for item in surge.xs(('likert dialogue', 'informative'),
                                                  level=['category', 'label']).index.tolist()}
    inter = data.dialogue_collection.annotation_dataframe()
    comparative_inter_dialogue_ids = {item[1] for item in inter.xs(('comparative', 'informative'),
                                                  level=['category', 'label']).index.tolist()}
    likert_inter_dialogue_ids = {item[1] for item in inter.xs(('likert dialogue', 'informative'),
                                                  level=['category', 'label']).index.tolist()}
    inter_missing = comparative_inter_dialogue_ids - comparative_surge_dialogue_ids
    inter_missing_l = likert_inter_dialogue_ids - likert_surge_dialogue_ids

    for m in inter_missing:
        d = data.dialogue_collection.dialogues[m]
        d.comparative_annotations.clear()



with open(f'{proj_dir}/data/data_student_external_comparative.json') as f:
    sec_json_data = json.load(f)
with open(f'{proj_dir}/data/data_student_external_likert.json') as f:
    sel_json_data = json.load(f)

from copy import deepcopy
json_data = {'student_external': deepcopy(sec_json_data['student_external_comparative'])}
for ident, d in json_data['student_external']['dialogues'].items():
    if ident in sel_json_data['student_external_likert']['dialogues']:
        d['likert_annotations'] = sel_json_data['student_external_likert']['dialogues'][ident]['likert_annotations']
json_data['student_external']['work_units'].update(sel_json_data['student_external_likert']['work_units'])
assert len(json_data['student_external']['work_units']) == (len(sel_json_data['student_external_likert']['work_units']) + len(sec_json_data['student_external_comparative']['work_units']))

data_student_ext = structure(json_data, StudentExtProject)

stu = data_student_ext.student_external.annotation_dataframe()
comparative_stu_dialogue_ids = {item[1] for item in stu.xs(('comparative', 'informative'),
                                              level=['category', 'label']).index.tolist()}
likert_stu_dialogue_ids = {item[1] for item in stu.xs(('likert dialogue', 'informative'),
                                              level=['category', 'label']).index.tolist()}
stu_missing = comparative_stu_dialogue_ids - comparative_surge_dialogue_ids
stu_missing_l = likert_stu_dialogue_ids - likert_surge_dialogue_ids

for m in stu_missing:
    d = data_student_ext.student_external.dialogues[m]
    d.comparative_annotations.clear()

for m in stu_missing_l:
    d = data_student_ext.student_external.dialogues[m]
    d.likert_annotations.clear()

##########
# Split into developer and non-developer
##########
def split_evaluation(workers_to_keep):
    split_project = StudentExtProject()
    split_project.student_external = Evaluation()
    split_project.student_external.work_units ={id: deepcopy(w) for id, w in data_student_ext.student_external.work_units.items() if w.worker_id in workers_to_keep}
    maintained_dialogue_ids = sorted({id for w in split_project.student_external.work_units.values() for id in w.dialogue_ids})
    split_project.student_external.dialogues = {id: deepcopy(data_student_ext.student_external.dialogues[id]) for id in maintained_dialogue_ids}
    for d in split_project.student_external.dialogues.values():
        for label, annotations in d.likert_annotations.items():
            to_keep = [annot for annot in annotations if annot.work_unit_id in split_project.student_external.work_units.keys()]
            d.likert_annotations[label] = to_keep
        for label, annotations in d.comparative_annotations.items():
            to_keep = [annot for annot in annotations if annot.work_unit_id in split_project.student_external.work_units.keys()]
            d.comparative_annotations[label] = to_keep
    return split_project

developers = {'espaek', 'cfcalvi', 'lssmit7'}
developer_ext = split_evaluation(developers)

all_workers = {w.worker_id for w in data_student_ext.student_external.work_units.values()}
non_developers = all_workers - developers
non_developer_ext = split_evaluation(non_developers)

# for d in data_student_ext.student_external.dialogues.values():
#     for label, annots in d.comparative_annotations.items():
#         workers = [data_student_ext.student_external.work_units[annot.work_unit_id].worker_id for annot in annots]
#         print(f"{d.dialogue_id}: {workers}")
#         break

x = 1

def print_stats(evaluation):
    # doubly annotated Likert
    singly_annotated_likert, doubly_annotated_likert = 0, 0
    singly_annotated_comparative, doubly_annotated_comparative = 0, 0
    for d in evaluation.dialogues.values():
        num_likert = len(d.likert_annotations['informative']) if len(d.likert_annotations) > 0 else 0
        if num_likert == 1:
            singly_annotated_likert += 1
        elif num_likert == 2:
            doubly_annotated_likert += 1

        num_comparative = len(d.comparative_annotations['informative']) if len(d.comparative_annotations) > 0 else 0
        if num_comparative == 1:
            singly_annotated_comparative += 1
        elif num_comparative == 2:
            doubly_annotated_comparative += 1
    print('Singly annotated likert: ', singly_annotated_likert)
    print('Doubly annotated likert: ', doubly_annotated_likert)
    print('Total likert', singly_annotated_likert + doubly_annotated_likert)
    print('Singly annotated comparative: ', singly_annotated_comparative)
    print('Doubly annotated comparative: ', doubly_annotated_comparative)
    print('Total comparative', singly_annotated_comparative + doubly_annotated_comparative)

print('Developers')
print_stats(developer_ext.student_external)

print()
print('Non Developers')
print_stats(non_developer_ext.student_external)
print()

print()
print('Surgers')
print_stats(data.surge_evaluation)
print()

class sym:

    category = 'category'
    label = 'label'
    bot = 'bot'
    bot_cmp = 'bot comp'
    item = 'item'
    stat = 'stat'


    def __call__(self):
        return [
            v for k, v in self.__class__.__dict__.items()
            if not k.startswith('__')
        ]

    def __iter__(self):
        return iter(self())

    def __contains__(self, item):
        return item in self()


class behavior(sym):
    antisocial = 'antisocial'
    common_contra = 'commonsense contradiction'
    partner_contra = 'partner contradiction'
    self_contra = 'self contradiction'
    ignore = 'ignore'
    incorrect_fact = 'incorrect fact'
    correct_fact = 'correct fact'
    irrelevant = 'irrelevant'
    redundant = 'redundant'
    lack_empathy = 'lack of empathy'
    uninterpretable = 'uninterpretable'
    empathetic = 'empathetic'
    follow_up = 'follow up'
    topic_switch = 'topic switch'
    life_info = 'life info'
    preference_info = 'preference info'
behavior = behavior()

class scale(sym):
    consistent = 'consistent'
    engaging = 'engaging'
    emotional = 'emotional'
    grammatical = 'grammatical'
    informative = 'informative'
    proactive = 'proactive'
    quality = 'quality'
    relevant = 'relevant'
scale = scale()

class category(sym):
    likert_dialogue = 'likert dialogue'
    likert_turn = 'likert turn'
    comparative = 'comparative'
    behavior = 'behavior'

class bot(sym):
    blender2 = 'blender2_3B'
    emora = 'emora'
    bart_fid_rag = 'bart_fid_rag_bcb'
    raranked_blender = 'rerank_blender'
    reranked_blender2 = 'rerank_blender2'
    cem = 'cem'
    dukenet = 'dukenet'
bot = bot()

class stat(sym):
    fleiss_kappa = "Fleiss' kappa"
    kripp_alpha = "Krippendorff's alpha"
    kend_tau = "Kendall's tau"
    mcfad_r2 = "McFadden's pseudo-R-squared"
    r2 = "R-Squared"
    ci_low = "CI low"
    ci_high = "CI high"
    proportion = 'proportion'
    mean = 'mean'
    n = 'n'
    likert_dialogue_quality = 'likert dialogue quality'
    likert_turn_quality = 'likert turn quality'
    p_of_f_test = 'P value of F-test'
    p_of_llr_test = 'P value of LLR-test'

class stage:
    annotation_pilots = 'annotation_pilots'
    annotation_pilots_onboarding = 'annotation_pilots_onboarding'
    bot_pilots = 'bot_pilots'
    extra_unused = 'extra_unused'
    dialogue_collection = 'dialogue_collection'
    student_evaluation = 'student_evaluation'
    student_onboarding = 'student_onboarding'
    student_gold_units = 'student_gold_units'
    mturk_evaluation = 'mturk_evaluation'
    mturk_onboarding = 'mturk_onboarding'
    mturk_gold_units = 'mturk_gold_units'
    surge_evaluation = 'surge_evaluation'
    surge_onboarding = 'surge_onboarding'
    surge_gold_units = 'surge_gold_units'
    expert_evaluation = 'expert_evaluation'




def bootstrap_ci(data, statistic_fn, n_resamples=10**3):
    wrapped_data = [dict(point=d) for d in data]
    statistic_fn_wrapper = lambda ds: statistic_fn([d['point'] for d in ds])
    result = retarded_bootstrap((wrapped_data,), statistic_fn_wrapper, vectorized=False, n_resamples=n_resamples)
    return result.confidence_interval

def fleiss_kappa(df, ci=False):
    """
    :param df: pandas dataframe: items x labeler: label
    :return: pandas series of kappa, CI low, CI high
    """
    def _fleiss_kappa(M):
          """
          See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
          :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
          :type M: numpy matrix
          """
          N, k = M.shape  # N is # of items, k is # of categories
          n_annotators = float(np.sum(M[0, :]))  # # of annotators
          p = np.sum(M, axis=0) / (N * n_annotators)
          P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
          Pbar = np.sum(P) / N
          PbarE = np.sum(p * p)
          if (1 - PbarE) == 0:
            kappa = np.nan
          else:
            kappa = (Pbar - PbarE) / (1 - PbarE)
          return kappa
    counts = df.stack().groupby(level=df.index.names).value_counts().unstack(fill_value=0)
    cnp = counts.to_numpy().astype(int)
    kappa = _fleiss_kappa(cnp)
    if ci:
        low, high = bootstrap_ci(cnp, lambda ds: _fleiss_kappa(np.array(ds)))
        n = len(df)
        result = {
            stat.fleiss_kappa: kappa,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.fleiss_kappa: kappa,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)


def krippendorfs_alpha(df, ci=True):
    """
    :param df: pandas dataframe: items x labeler: label
    :return:
    """
    ratings = df.to_numpy()
    ka = lambda x: krippendorff.alpha(x.T, level_of_measurement='ordinal')
    try:
        alpha = ka(ratings)
    except AssertionError:
        alpha = None
    if ci:
        try:
            low, high = bootstrap_ci(ratings, lambda x: ka(np.array(x)))
        except AssertionError:
            low, high = None, None
        result = {
            stat.kripp_alpha: alpha,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.kripp_alpha: alpha,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)


def mean_and_ci(df: pd.DataFrame):
    vals = df.to_numpy()
    mean = vals.mean()
    def t_conf_int(data, alpha=0.05):
        mean = sum(data) / len(data)
        stderr = stats.sem(data)
        return stats.t.interval(
            alpha=(1 - alpha),
            df=len(data) - 1,
            loc=mean,
            scale=stderr
        )
    (low,), (high,) = t_conf_int(vals)
    result = {stat.mean: mean, stat.ci_low: low, stat.ci_high: high, stat.n: len(df)}
    return pd.Series(result.values(), result)


def prop_and_ci(df: pd.DataFrame):
    vals = df.to_numpy()
    positives = vals.sum()
    total = len(vals)
    prop = positives / total
    low, high = sm.stats.proportion_confint(positives, total, method='wilson')
    if not isinstance(low, float):
        low, = low
    if not isinstance(high, float):
        high, = high
    result = {stat.proportion: prop, stat.ci_low: low, stat.ci_high: high, stat.n: len(df)}
    return pd.Series(result.values(), result)


def to_file(f):
    def fn_to_file(*args, load=None, reload=None, **kwargs):
        if load:
            return pd.read_pickle(load)
        result = f(*args, **kwargs)
        if reload:
            result.to_pickle(reload)
            return pd.read_pickle(reload)
        return result
    return fn_to_file


def prettify(df, float_prec=None, col_types=None, sort_by=None, to_csv=None, index=True, header=True):
    if col_types:
        for col, type in col_types.items():
            df[col] = df[col].astype(type)
    if sort_by:
        df.sort_values(sort_by, ascending=False, inplace=True)
    if float_prec:
        df = df.round(float_prec)
    if to_csv:
        df.to_csv(to_csv, float_format=f"%.{float_prec}f" if float_prec is not None else None, header=header, index=index)
    return df


@to_file
def across_evaluations(annotations, evaluation_fn):
    """
    :param annotations: iterable of annotations df to apply evaluation_fn to
    :param evaluation_fn: function (input is annotations df, output is results df)
    :return: results dataframe where first index level codes which evaluation (integer id)
    """
    results = [evaluation_fn(annotation) for annotation in annotations]
    all_results = pd.concat(results, keys=range(len(results)))
    all_results.index.set_names('round', level=0, inplace=True)
    return all_results

def get_example(
        evaluation,
        category,
        label,
        mark=None,
        bot=None,
        context=0,
        seed=123,
        annotations: pd.DataFrame = None,
        item_id=None
):
    if annotations is None:
        annotations = evaluation.annotation_dataframe()
    if item_id:
        eid = item_id
    else:
        labels = annotations.xs((category, label), level=(1, 2)).reset_index()
        options = labels[labels[0] == mark]
        if bot:
            options = options[options[sym.bot] == bot]
        try:
            example = options.sample(1, random_state=seed)
        except ValueError:
            return f'No samples for {category} {label} {mark} {bot}\n'
        eid = example[sym.item].item()
    if isinstance(eid, tuple):
        did, tid = eid
        turns = evaluation.dialogues[did].turns[max(0, tid-context):tid+1]
        botstring = '' if not bot else f'{bot}~~~\n'
        contextstring = ''.join((
            (
                f'User:  {turn.user_turn}\n'
                f'Sys:   {turn.bot_turn}\n'
            )
            for turn in turns[:-1]
        ))
        turn = turns[-1]
        if mark is None:
            mark = annotations.xs((category, label, eid), level=(1, 2, 3))[0].item()
        turnstring = (
            f'User:  {turn.user_turn}\n'
            f'Sys:   {turn.bot_turn}\n'
            f'Label: {label} = {mark}\n'
        )
        display = botstring + contextstring + turnstring
    else:
        dialogue = evaluation.dialogues[eid]
        turns = [
            turn
            for turn_pair in dialogue.turns
            for turn in (turn_pair.user_turn, turn_pair.bot_turn)
        ]
        if mark is None:
            mark = annotations.xs((category, label, eid), level=(1, 2, 3))[0].item()
        display = '\n'.join([f'{dialogue.bot}~~~', *turns, f'Label: {label} = {mark}\n'])
    print(display)
    return eid

@to_file
def interactor_summary_stats(evaluation: edd.Evaluation):
    num_dialogues = len(evaluation.dialogues)
    mean_turns = (
        sum((
            2*len(d.turns)
            for d in evaluation.dialogues.values()
        ))
        / num_dialogues
    )
    user_turn_len = (
        sum((
            len(nltk.word_tokenize(t.user_turn))
            for d in evaluation.dialogues.values()
            for t in d.turns
        ))
        / sum((
            len(d.turns)
            for d in evaluation.dialogues.values()
        ))
    )
    num_interactors = len({
        unit.worker_id
        for unit in evaluation.work_units.values()
    })
    summary = {
        'dialogues': num_dialogues,
        'mean turns': mean_turns,
        'user turn length': user_turn_len,
        'interactors': num_interactors,
    }
    return pd.DataFrame(summary.values(), summary)

@to_file
def screening_rates_by_label(evaluation: edd.OnboardingEvaluation):
    perfs = {}
    workers_passed = {}
    workers_attempted = {}
    for did, dialogue in evaluation.dialogues.items():
        for attempt in dialogue.attempts:
            work_unit = evaluation.work_units[attempt.work_unit_id]
            round = int(did.split('_')[-1])
            task = work_unit.task
            labels = work_unit.labels
            num_mistakes = len(attempt.mistakes)
            worker = work_unit.worker_id
            accuracy = attempt.performance
            perfs.setdefault(task, []).append((num_mistakes, accuracy))
            workers_attempted.setdefault(task, set()).add(worker)
            if attempt.passed:
                workers_passed.setdefault(task, set()).add(worker)
    screening = {}
    for task, ls in perfs.items():
        mistakes, accuracies = zip(*ls)
        avg_m = sum(mistakes) / len(mistakes)
        avg_a = (
            sum(accuracies) / len(accuracies)
            if all((a is not None for a in accuracies)) else None
        )
        n = len(mistakes)
        attempted = len(workers_attempted.get(task, ()))
        passed = len(workers_passed.get(task, ()))
        screening[task] = {
            'attempted': attempted, 'passed': passed,
            'mistakes': avg_m, 'accuracy': avg_a, 'n': n
        }
    return pd.DataFrame(screening.values(), screening)

def get_doubly_annotated(annotations, k=2, dropna=True):
    doubly_annotated = annotations.iloc[:,:k]
    if dropna:
        doubly_annotated = doubly_annotated.dropna()
    return doubly_annotated

@to_file
def agreement_dataframe(annotations, ci=True, k=2, dropna=True):
    doubly_annotated = get_doubly_annotated(annotations, k, dropna)
    label_groups = doubly_annotated.groupby(level=[sym.category, sym.label])
    agreements = label_groups.apply(krippendorfs_alpha, ci=ci)
    return agreements

def get_correlation(df, method):
    if method == 'spearman':
        result = spearmanr(df)
    elif method == 'pearson':
        result = pearsonr(df[df.columns[0]], df[df.columns[1]])
    return pd.DataFrame({f'{method} correlation': [result[0]], f'{method} pvalue': [result[1]]})

def correlation_dataframe(annotations, method, k=2, dropna=True):
    doubly_annotated = annotations.iloc[:,:k]
    if dropna:
        doubly_annotated = doubly_annotated.dropna()
    label_groups = doubly_annotated.groupby(level=[sym.category, sym.label])
    correlations = label_groups.apply(lambda x: get_correlation(x, method))
    correlations = correlations.droplevel(2)
    return correlations


def get_singly_annotated(df: pd.DataFrame, seed=123):
    if len(df.columns) == 1:
        return df.astype(int)
    previous_state = random.getstate()
    random.seed(seed)

    df = df.iloc[:,:2]
    mask = df[1].isna()
    singly_annotated = df.iloc[:,0][mask]
    doubly_annotated = df[~mask]
    reflections = {}
    selection = []
    for i in range(len(doubly_annotated)):
        item = doubly_annotated.index.values[i][-1]
        item_code = frozenset(item)
        selection.append(reflections.setdefault(item_code, random.randint(0, 1)))
    indices = list(range(len(doubly_annotated)))
    select_annotated = doubly_annotated.values[indices, selection]
    select_annotated = pd.DataFrame(select_annotated, index=doubly_annotated.index)
    annotations = pd.concat((singly_annotated, select_annotated))
    random.setstate(previous_state)
    return annotations.astype(int)


@to_file
def aggregate_comparative(annotations):
    single_annotated = get_singly_annotated(annotations, seed=123)
    prop_dfs = []
    for cmp, cmp_label in {-1: 'lose', 0: 'tie', 1: 'win'}.items():
        annotated = single_annotated == cmp
        annotated = annotated.astype(int)
        groups = annotated.groupby(level=[sym.bot, sym.bot_cmp, sym.label])
        props = groups.apply(prop_and_ci)
        props.rename(columns={stat.proportion: cmp_label}, inplace=True)
        prop_dfs.append(props)
    result = pd.concat(prop_dfs, axis=1)
    prop_dfs = []
    for cmp, cmp_label in {-1: 'lose', 0: 'tie', 1: 'win'}.items():
        annotated = single_annotated == cmp
        annotated = annotated.astype(int)
        groups = annotated.groupby(level=[sym.bot, sym.label])
        props = groups.apply(prop_and_ci)
        props.rename(columns={stat.proportion: cmp_label}, inplace=True)
        prop_dfs.append(props)
    result_vs_all = pd.concat(prop_dfs, axis=1)
    others_idx = {sym.bot_cmp: 'others'}
    result_vs_all = result_vs_all.assign(**others_idx)
    levels = [sym.bot, sym.bot_cmp, sym.label]
    result_vs_all = result_vs_all.set_index(sym.bot_cmp, append=True)
    result_vs_all = result_vs_all.reset_index().set_index(levels)
    result = pd.concat((result_vs_all, result))
    return result

def aggregate_likert_ratings(annotations, category, load=None, reload=None, bot_groups=True):
    if load:
        return pd.read_csv(load)
    single_annotated = get_singly_annotated(annotations)
    likert_annotations = single_annotated.xs(category, level=sym.category)
    if bot_groups:
        label_groups = likert_annotations.groupby(level=[sym.bot, sym.label])
    else:
        label_groups = likert_annotations.groupby(level=[sym.label])
    means = label_groups.apply(mean_and_ci)
    if reload:
        means.to_csv(reload)
    return means

def aggregate_behavior_rates(annotations, load=None, reload=None):
    if load:
        return pd.read_csv(load)
    single_annotated = get_singly_annotated(annotations)
    behavior_annotations = single_annotated.xs(category.behavior, level=sym.category)
    label_groups = behavior_annotations.groupby(level=[sym.bot, sym.label])
    means = label_groups.apply(prop_and_ci)
    if reload:
        means.to_csv(reload)
    return means

def p_vals(df: pd.DataFrame, test='t', downsample=None, replace=False, deterministic=True):
    """
    :param df: (bot, data point) x 1 -> score
    :param test: statistical test function (t for t test, p for prop test, s for sign test)
    :param downsample: number of samples ber bot to subsample without replacement for the analysis
    :return: p values of test on each bot pair (pd.Series)
    """
    seed = None
    if deterministic:
        seed = 123
        random.seed(seed)
    bots = sorted(set(df.index.get_level_values(0)))
    num_bots = len(bots)
    bot_pairs = list(combinations(bots, 2))
    result = {}
    for ba, bb in bot_pairs:
        if test == 't':
            a = downsample_dials(df.xs(ba), downsample, replace).to_numpy().squeeze()
            b = downsample_dials(df.xs(bb), downsample, replace).to_numpy().squeeze()
            t, p = ttest_ind(a, b, equal_var=False)
        elif test == 'p':
            a = downsample_dials(df.xs(ba), downsample, replace).to_numpy().squeeze()
            b = downsample_dials(df.xs(bb), downsample, replace).to_numpy().squeeze()
            sums = sum(a), sum(b)
            z, p = proportions_ztest(count=[
                *sums
            ], nobs=[
                len(a), len(b)
            ])
        elif test == 's':
            # sign test
            comp_data = df.xs((ba, bb), level=[sym.bot, sym.bot_cmp])
            if downsample:
                comp_data = comp_data.sample(downsample, replace=replace, random_state=seed)
            a = comp_data.to_numpy().squeeze() == 1
            b = comp_data.to_numpy().squeeze() == -1
            p = binom_test(sum(a), sum(a)+sum(b), p=0.5)
        else:
            raise ValueError('invalid arg for param "test"')
        result[(ba, bb)] = p
    result_series = pd.Series(result.values(), result)
    return result_series

def p_values_comparing_bots(evaluation, downsample=None, replace=False, exclude_comparative=False, deterministic=True, mean_annotations=None, comp_annotations=None):
    if mean_annotations is None:
        annotations = get_singly_annotated(evaluation.annotation_dataframe(), seed=123)
        mean_annotations = annotations.drop(
            index=category.behavior, level=sym.category, errors='ignore'
        ).drop(
            index=category.comparative, level=sym.category
        ).drop(
            index=category.likert_turn, level=sym.category, errors='ignore'
        )
    mean_ps = mean_annotations.groupby(
        [sym.category, sym.label]
    ).apply(lambda x: p_vals(x, test='t', downsample=downsample, replace=replace, deterministic=deterministic))
    if not exclude_comparative:
        if comp_annotations is None:
            comp_annotations = get_singly_annotated(evaluation.comparative_annotation_dataframe(), seed=123)
        comp_groups = comp_annotations.groupby(sym.label)
        comp_ps = comp_groups.apply(lambda x: p_vals(x, test='s', downsample=downsample, replace=replace, deterministic=deterministic))
        comp_ps = pd.concat({category.comparative: comp_ps}, names=[sym.category])
        result = pd.concat([mean_ps, comp_ps], axis=0)
    else:
         result = pd.concat([mean_ps], axis=0)
    return result

def p_values_comparing_bots_comparative(evaluation, downsample=None, replace=False, deterministic=True, comp_annotations=None):
    if comp_annotations is None:
        comp_annotations = get_singly_annotated(evaluation.comparative_annotation_dataframe(), seed=123)
    comp_groups = comp_annotations.groupby(sym.label)
    comp_ps = comp_groups.apply(
        lambda x: p_vals(x, test='s', downsample=downsample, replace=replace, deterministic=deterministic))
    comp_ps = pd.concat({category.comparative: comp_ps}, names=[sym.category])
    result = pd.concat([comp_ps], axis=0)
    return result

__all__ = [

    # the data
    'data',
    'data_student_ext',
    'developer_ext',
    'non_developer_ext',

    # symbols
    'sym',
    'scale',
    'behavior',
    'category',
    'stage',
    'bot',
    'stat',

    # stats
    'fleiss_kappa',
    'krippendorfs_alpha',
    'mean_and_ci',
    'prop_and_ci',
    'p_values_comparing_bots',
    'p_values_comparing_bots_comparative',

    # utils
    'to_file',
    'prettify',
    'across_evaluations',
    'get_example',
    'interactor_summary_stats',
    'screening_rates_by_label',
    'agreement_dataframe',
    'get_singly_annotated',
    'get_doubly_annotated',
    'aggregate_comparative',
    'aggregate_likert_ratings',
    'aggregate_behavior_rates',
    'correlation_dataframe'

]


if __name__ == '__main__':
    pass
    # td = {
    #     ('emora', 1):   {'x': 2, 'y': 9, 'q': 1},
    #     ('emora', 2):   {'x': 3, 'y': 5, 'q': 2},
    #     ('emora', 3):   {'x': 5, 'y': 1, 'q': 3},
    #     ('blender', 1): {'x': 5, 'y': 2, 'q': 1},
    #     ('blender', 2): {'x': 4, 'y': 3, 'q': 2},
    #     ('blender', 3): {'x': 5, 'y': 4, 'q': 2},
    #     ('blender', 4): {'x': 4, 'y': 5, 'q': 3}
    # }
    # tdf = pd.DataFrame(td.values(), td)
    #
    # result = t_tests(tdf)
    # print(result)


