from numpy import mean, var, sqrt, abs
import numpy as np

def cohensd(d1, d2):
    if isinstance(d1, list):
        d1 = np.array(d1)
    if isinstance(d2, list):
        d2 = np.array(d2)
    if len(d1.shape) == 1:
        n1 = d1.shape[0]
        d1 = d1.reshape((1, -1))
    elif len(d1.shape) == 2:
        n1 = d1.shape[1]
    if len(d2.shape) == 1:
        n2 = d2.shape[0]
        d2 = d2.reshape((1, -1))
    elif len(d2.shape) == 2:
        n2 = d2.shape[1]
    s1, s2 = var(d1, ddof=1, axis=1), var(d2, ddof=1, axis=1)
    s1_ = (n1 - 1) * s1
    s2_ = (n2 - 1) * s2
    s1_s2_ = s1_ + s2_
    s1_s2__ = s1_s2_ / (n1 + n2 - 2)
    s = sqrt(s1_s2__) # pooled standard deviation
    u1, u2 = mean(d1, axis=1), mean(d2, axis=1)
    return (u1 - u2) / s

def effect_size_likert(group1_bot1, group1_bot2, group2_bot1, group2_bot2, axis=None):
    group1_effectsize = cohensd(group1_bot1, group1_bot2)
    group2_effectsize = cohensd(group2_bot1, group2_bot2)
    diff = group1_effectsize - group2_effectsize
    return abs(diff)




def win_proportion(arr):
    wins = np.count_nonzero(arr == 1, axis=1)
    losses = np.count_nonzero(arr == -1, axis=1)
    win_props = wins / (wins + losses)
    return win_props

def cohensh(d1, d2):
    if isinstance(d1, list):
        d1 = np.array(d1)
    if isinstance(d2, list):
        d2 = np.array(d2)
    if len(d1.shape) == 1:
        n1 = d1.shape[0]
        d1 = d1.reshape((1, -1))
    elif len(d1.shape) == 2:
        n1 = d1.shape[1]
    if len(d2.shape) == 1:
        n2 = d2.shape[0]
        d2 = d2.reshape((1, -1))
    elif len(d2.shape) == 2:
        n2 = d2.shape[1]
    p1 = win_proportion(d1)
    p2 = win_proportion(d2)
    return 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))

def effect_size_comparative(group1_bot1, group1_bot2, group2_bot1, group2_bot2, axis=None):
    group1_effectsize = cohensh(group1_bot1, group1_bot2)
    group2_effectsize = cohensh(group2_bot1, group2_bot2)
    diff = group1_effectsize - group2_effectsize
    return np.abs(diff)