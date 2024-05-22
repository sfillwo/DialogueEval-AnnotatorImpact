# Chat-Oriented Dialogue: Human Evaluator Group Evaluations

This repo contains the data and analysis scripts for the paper: 

[Exploring the Impact of Human Evaluator Group on Chat-Oriented Dialogue Evaluation](https://arxiv.org/pdf/2309.07998)

**Abstract:** Human evaluation has been widely accepted as
the standard for evaluating chat-oriented dialogue systems. However, there is a significant
variation in previous work regarding who gets
recruited as evaluators. Evaluator groups such
as domain experts, university students, and professional annotators have been used to assess
and compare dialogue systems, although it is
unclear to what extent the choice of an evaluator group can affect results. This paper analyzes
the evaluator group impact on dialogue system
evaluation by testing 4 state-of-the-art dialogue
systems using 4 distinct evaluator groups. Our
analysis reveals a robustness towards evaluator
groups for Likert evaluations that is not seen for
Pairwise, with only minor differences observed
when changing evaluator groups. Furthermore,
two notable limitations to this robustness are
observed, which reveal discrepancies between
evaluators with different levels of chatbot expertise and indicate that evaluator objectivity is
beneficial for certain dialogue metrics.

`agreements.py` calculates Dialogue Score Agreement using Inter-Annotator agreement measure Krippendorff's alpha.

`effect_size_difference_comparative.py` and `effect_size_difference_likert.py` calculate the Bot-Level Performance Agreement using effect sizes differences.