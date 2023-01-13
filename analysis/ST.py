from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats


def stat_test(data):
    flat = [item for sublist in data for item in sublist]
    groups = []

    for i in range(5):
        for j in range(len(data[i])):
            groups.append(i)

    f_statistic, p_value = stats.f_oneway(*data)
    result = pairwise_tukeyhsd(flat, groups, 0.05)

    return f_statistic, p_value, result
