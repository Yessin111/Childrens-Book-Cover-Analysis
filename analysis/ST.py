import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy.stats import chisquare
from statsmodels.stats.libqsturng import psturng


def anova_turkey(data, n_ages):
    flat = [item for sublist in data for item in sublist]
    groups = []

    for i in range(n_ages):
        for j in range(len(data[i])):
            groups.append(i)

    f_statistic, p_value = stats.f_oneway(*data)
    hsd = pairwise_tukeyhsd(flat, groups, 0.05)
    result = psturng(np.abs(hsd.meandiffs / hsd.std_pairs), len(hsd.groupsunique), hsd.df_total)

    return f_statistic, p_value, result


def chi(freq, count, n_ages, num_topics):
    p_values = []
    for i in range(n_ages):
        current = []
        for j in range(num_topics):
            if count[i] != 0:
                current.append((freq[i][j]/count[i]) * (sum(count)/n_ages))
            else:
                current.append(0)

        ex_freq = [sum(freq[i])/len(freq[i])] * len(freq[i])
        p_values.append(chisquare(freq[i], ex_freq)[1])

    return p_values
