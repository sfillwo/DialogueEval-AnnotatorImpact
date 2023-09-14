import statistics

from utilities.analysis import *
import pandas as pd
from collections import defaultdict

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def get_stats(worker_dict):
    counts = worker_dict.values()
    return len(counts), min(counts), statistics.median(counts), max(counts)

interactive_workers = defaultdict(int)
interactive_workers_by_type = {}
for unit in data.dialogue_collection.work_units.values():
    interactive_workers_by_type.setdefault('likert', defaultdict(int))[unit.worker_id] += 2
    interactive_workers_by_type.setdefault('comparative', defaultdict(int))[unit.worker_id] += 1
    interactive_workers[unit.worker_id] += 3
interactive_stats = get_stats(interactive_workers)
interactive_likert_stats = get_stats(interactive_workers_by_type['likert'])
interactive_comparative_stats = get_stats(interactive_workers_by_type['comparative'])

external_workers = defaultdict(int)
external_workers_by_type = {}
for unit in data.surge_evaluation.work_units.values():
    if unit.worker_id != '{{user_id}}' and unit.task in {'comparative', 'likert'}:
        external_workers_by_type.setdefault(unit.task, defaultdict(int))[unit.worker_id] += 1
        external_workers[unit.worker_id] += 1
external_stats = get_stats(external_workers)
external_likert_stats = get_stats(external_workers_by_type['likert'])
external_comparative_stats = get_stats(external_workers_by_type['comparative'])

external_student_workers = defaultdict(int)
external_student_workers_by_type = {}
for unit in data_student_extcomp.student_external_comparative.work_units.values():
    if unit.worker_id != '{{user_id}}' and unit.task in {'comparative', 'likert'}:
        external_student_workers_by_type.setdefault(unit.task, defaultdict(int))[unit.worker_id] += 1
        external_student_workers[unit.worker_id] += 1
external_student_stats = get_stats(external_student_workers)
external_student_comparative_stats = get_stats(external_student_workers_by_type['comparative'])

stats_df = pd.DataFrame({'interactive': interactive_stats,
                         'interactive likert': interactive_likert_stats,
                         'interactive comparative': interactive_comparative_stats,
                         'external': external_stats,
                         'external likert': external_likert_stats,
                         'external comparative': external_comparative_stats,
                         'external student': external_student_stats,
                         'external student comparative': external_student_comparative_stats})
stats_df.index = ['#', 'Min', 'Median', 'Max']
print()
print(stats_df)
stats_df.to_csv('outputs/csv/annotator_statistics.csv')