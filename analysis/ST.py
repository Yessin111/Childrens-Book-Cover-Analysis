from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy import chisquare


def anova_turkey(data):
    flat = [item for sublist in data for item in sublist]
    groups = []

    for i in range(5):
        for j in range(len(data[i])):
            groups.append(i)

    f_statistic, p_value = stats.f_oneway(*data)
    result = pairwise_tukeyhsd(flat, groups, 0.05)

    return f_statistic, p_value, result


def chi(freq, count, n_ages, num_topics):
    p_values = []
    for i in range(num_topics):
        current = []
        for j in range(n_ages):
            current.append((freq[j][i]/count[i]) * (sum(count)/n_ages))

        ex_freq = [sum(freq)/n_ages]
        p_values.append(chisquare(freq, ex_freq)[1])

    return p_values
