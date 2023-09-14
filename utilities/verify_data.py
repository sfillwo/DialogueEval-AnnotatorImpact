import matplotlib.pyplot as plt

from analysis import *
from graphing import *
from utils import *
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import binom_test
from itertools import combinations
import random

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

for bot in interactive_annotations.index.unique('bot'):
    assert interactive_annotations.xs((bot, 'likert dialogue', 'informative'),
                                      level=['bot', 'category', 'label']).size == 100
    assert interactive_annotations.xs((bot, 'comparative', 'informative'),
                                      level=['bot', 'category', 'label']).size == 96

    assert external_annotations.xs((bot, 'likert dialogue', 'informative'),
                                      level=['bot', 'category', 'label']).size == 100
    assert external_annotations.xs((bot, 'comparative', 'informative'),
                                      level=['bot', 'category', 'label']).size == 96

    assert ux.xs((bot, 'comparative', 'informative'),
                                      level=['bot', 'category', 'label']).size == 96


