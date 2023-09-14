import pandas as pd
from utilities.analysis import *

surge_timings = data.surge_evaluation.timing_dataframe().drop(
    index=category.behavior, level=sym.category, errors='ignore'
).drop(
    index=category.likert_turn, level=sym.category, errors='ignore'
)
interactive_timings = data.dialogue_collection.interactive_timing_dataframe()
student_ext_comp_timings = data_student_extcomp.student_external_comparative.timing_dataframe()
all_tasks_timings = pd.concat({'sx': surge_timings, 'ui': interactive_timings, 'ux': student_ext_comp_timings}, names=['eval'])

all_tasks_timings_minutes = all_tasks_timings / 60
grouped_timings_per_task = all_tasks_timings_minutes.groupby(level=['eval', 'category', 'labels'])

med_timings_per_task = grouped_timings_per_task.median()
med_timings_per_task.columns = ['completion time (minutes)']
std_per_task = grouped_timings_per_task.std()
std_per_task.columns = ['std (minutes)']
statistics = pd.concat([med_timings_per_task, std_per_task], axis=1)
statistics['cv'] = statistics['std (minutes)'] / statistics['completion time (minutes)']

print('Completion time statistics')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(statistics)

# grouped_timings_per_task.hist()
# plt.show()

