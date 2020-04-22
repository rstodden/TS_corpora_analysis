import pandas as pd
from scipy import stats
import numpy as np
import math
import os
import sys
import json, csv
import itertools as it
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scikit_posthocs
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import OrderedDict
from sklearn.metrics import r2_score
from scipy.stats import distributions
from scipy.stats.stats import find_repeats
import warnings

def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
			 alternative="two-sided"):
	"""
	scipy stats function https://github.com/scipy/scipy/blob/v1.2.1/scipy/stats/morestats.py#L2709-L2806
	Calculate the Wilcoxon signed-rank test.

	The Wilcoxon signed-rank test tests the null hypothesis that two
	related paired samples come from the same distribution. In particular,
	it tests whether the distribution of the differences x - y is symmetric
	about zero. It is a non-parametric version of the paired T-test.

	Parameters
	----------
	x : array_like
		Either the first set of measurements (in which case `y` is the second
		set of measurements), or the differences between two sets of
		measurements (in which case `y` is not to be specified.)  Must be
		one-dimensional.
	y : array_like, optional
		Either the second set of measurements (if `x` is the first set of
		measurements), or not specified (if `x` is the differences between
		two sets of measurements.)  Must be one-dimensional.
	zero_method : {'pratt', 'wilcox', 'zsplit'}, optional
		The following options are available (default is 'wilcox'):

		  * 'pratt': Includes zero-differences in the ranking process,
			but drops the ranks of the zeros, see [4]_, (more conservative).
		  * 'wilcox': Discards all zero-differences, the default.
		  * 'zsplit': Includes zero-differences in the ranking process and
			split the zero rank between positive and negative ones.
	correction : bool, optional
		If True, apply continuity correction by adjusting the Wilcoxon rank
		statistic by 0.5 towards the mean value when computing the
		z-statistic.  Default is False.
	alternative : {"two-sided", "greater", "less"}, optional
		The alternative hypothesis to be tested, see Notes. Default is
		"two-sided".

	Returns
	-------
	statistic : float
		If `alternative` is "two-sided", the sum of the ranks of the
		differences above or below zero, whichever is smaller.
		Otherwise the sum of the ranks of the differences above zero.
	pvalue : float
		The p-value for the test depending on `alternative`.

	See Also
	--------
	kruskal, mannwhitneyu

	Notes
	-----
	The test has been introduced in [4]_. Given n independent samples
	(xi, yi) from a bivariate distribution (i.e. paired samples),
	it computes the differences di = xi - yi. One assumption of the test
	is that the differences are symmetric, see [2]_.
	The two-sided test has the null hypothesis that the median of the
	differences is zero against the alternative that it is different from
	zero. The one-sided test has the null hypothesis that the median is
	positive against the alternative that it is negative
	(``alternative == 'less'``), or vice versa (``alternative == 'greater.'``).

	The test uses a normal approximation to derive the p-value (if
	``zero_method == 'pratt'``, the approximation is adjusted as in [5]_).
	A typical rule is to require that n > 20 ([2]_, p. 383). For smaller n,
	exact tables can be used to find critical values.

	References
	----------
	.. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
	.. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
	.. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
	   Rank Procedures, Journal of the American Statistical Association,
	   Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
	.. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
	   Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
	.. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
	   Sampling Distribution When Zero Differences are Present,
	   Journal of the American Statistical Association, Vol. 62, 1967,
	   pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`

	Examples
	--------
	In [4]_, the differences in height between cross- and self-fertilized
	corn plants is given as follows:

	>>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

	Cross-fertilized plants appear to be be higher. To test the null
	hypothesis that there is no height difference, we can apply the
	two-sided test:

	>>> from scipy.stats import wilcoxon
	>>> w, p = wilcoxon(d)
	>>> w, p
	(24.0, 0.04088813291185591)

	Hence, we would reject the null hypothesis at a confidence level of 5%,
	concluding that there is a difference in height between the groups.
	To confirm that the median of the differences can be assumed to be
	positive, we use:

	>>> w, p = wilcoxon(d, alternative='greater')
	>>> w, p
	(96.0, 0.020444066455927955)

	This shows that the null hypothesis that the median is negative can be
	rejected at a confidence level of 5% in favor of the alternative that
	the median is greater than zero. The p-value based on the approximation
	is within the range of 0.019 and 0.054 given in [2]_.
	Note that the statistic changed to 96 in the one-sided case (the sum
	of ranks of positive differences) whereas it is 24 in the two-sided
	case (the minimum of sum of ranks above and below zero).

	"""

	if zero_method not in ["wilcox", "pratt", "zsplit"]:
		raise ValueError("Zero method should be either 'wilcox' "
						 "or 'pratt' or 'zsplit'")

	if alternative not in ["two-sided", "less", "greater"]:
		raise ValueError("Alternative must be either 'two-sided', "
						 "'greater' or 'less'")

	if y is None:
		d = np.asarray(x)
		if d.ndim > 1:
			raise ValueError('Sample x must be one-dimensional.')
	else:
		x, y = map(np.asarray, (x, y))
		if x.ndim > 1 or y.ndim > 1:
			raise ValueError('Samples x and y must be one-dimensional.')
		if len(x) != len(y):
			raise ValueError('The samples x and y must have the same length.')
		d = x - y

	if zero_method in ["wilcox", "pratt"]:
		n_zero = np.sum(d == 0, axis=0)
		if n_zero == len(d):
			raise ValueError("zero_method 'wilcox' and 'pratt' do not work if "
							 "the x - y is zero for all elements.")

	if zero_method == "wilcox":
		# Keep all non-zero differences
		d = np.compress(np.not_equal(d, 0), d, axis=-1)

	count = len(d)
	if count < 10:
		warnings.warn("Sample size too small for normal approximation.")

	r = stats.rankdata(abs(d))
	r_plus = np.sum((d > 0) * r, axis=0)
	r_minus = np.sum((d < 0) * r, axis=0)

	if zero_method == "zsplit":
		r_zero = np.sum((d == 0) * r, axis=0)
		r_plus += r_zero / 2.
		r_minus += r_zero / 2.

	# return min for two-sided test, but r_plus for one-sided test
	# the literature is not consistent here
	# r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
	# i.e. the sum of the ranks, so r_minus and the min can be inferred
	# (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
	# [3] uses the r_plus for the one-sided test, keep min for two-sided test
	# to keep backwards compatibility
	if alternative == "two-sided":
		T = min(r_plus, r_minus)
	else:
		T = r_plus
	mn = count * (count + 1.) * 0.25
	se = count * (count + 1.) * (2. * count + 1.)

	if zero_method == "pratt":
		r = r[d != 0]
		# normal approximation needs to be adjusted, see Cureton (1967)
		mn -= n_zero * (n_zero + 1.) * 0.25
		se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

	replist, repnum = find_repeats(r)
	if repnum.size != 0:
		# Correction for repeated elements.
		se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

	se = np.sqrt(se / 24)

	# apply continuity correction if applicable
	d = 0
	if correction:
		if alternative == "two-sided":
			d = 0.5 * np.sign(T - mn)
		elif alternative == "less":
			d = -0.5
		else:
			d = 0.5

	# compute statistic and p-value using normal approximation
	z = (T - mn - d) / se
	if alternative == "two-sided":
		prob = 2. * distributions.norm.sf(abs(z))
	elif alternative == "greater":
		# large T = r_plus indicates x is greater than y; i.e.
		# accept alternative in that case and return small p-value (sf)
		prob = distributions.norm.sf(z)
	else:
		prob = distributions.norm.cdf(z)

	return T, prob, z


def get_effect_size_text(effect_size):
	if effect_size == None:
		effect_name = "unknown"
	elif 0.1 <= effect_size < 0.25:
		effect_name = "uphill weak"
	elif 0.25 <= effect_size < 0.4:
		effect_name = "uphill moderate"
	elif effect_size >= 0.4:
		effect_name = "uphill strong"
	elif -0.1 >= effect_size > -0.25:
		effect_name = "downhill weak"
	elif -0.25 >= effect_size > -0.4:
		effect_name = "downhill moderate"
	elif effect_size <= -0.4:
		effect_name = "downhill strong"
	else:
		effect_name = "unsure"
	return effect_name


def get_p_value_stars(p_value):
	if p_value <= 0.01:
		return "***"
	elif p_value <= 0.05:
		return "**"
	elif p_value <= 0.1:
		return "*"
	else:
		return ""

def get_result_sent(test_name, feature_name, corpus_name, p_value, n_complex, avg_complex, sd_complex, n_simple, avg_simple, sd_simple, df, t_value, effect_size, p_threshold=0.05, only_relevant=False):
	effect_name = get_effect_size_text(effect_size)
	if 0 <= p_value <= p_threshold:
		is_significant = "a"
		p_value_text = "p<="+str(p_threshold)
	else:
		is_significant = "no"
		p_value_text = "p>"+str(p_threshold)
	if test_name == "No test" or effect_size == None:
		return "The average of {} for complex sentences is {} (SD={}, n={}) and for simple sentences {} (SD={}).".format(feature_name,  round(avg_complex,2), round(sd_complex, 2), n_complex, round(avg_simple, 2), round(sd_simple, 2))
	if only_relevant:
		if p_value > p_threshold or effect_size == None or effect_size < 0.1:
			return None
	return "A {} was conducted to compare {} in the {} corpus. " \
	"There is {} significant ({}) difference in the scores for complex (n={}, M={}, SD={}) and " \
	"simplified (n={}, M={}, SD={}) sentences, t({})={}. " \
	"These results that the simplification level has a {} effect (r={}) on {}.\n".format(test_name, feature_name,
																					  corpus_name, is_significant,
																					  p_value_text, n_complex, round(avg_complex,2),
																					  round(sd_complex,2), n_simple, round(avg_simple,2),
																					  round(sd_simple,2), df, round(t_value,2),
																					  effect_name, round(effect_size,2),
																					  feature_name)


def get_variable_names(col_names, feat_dict_path="feature_dict_checked.json", comparable=False, paired=True, difference=False):
	if comparable:
		return sorted(list(set(["_".join(col.split("_")[:-1]) for col in col_names if col.endswith("_complex") or col.endswith("_simple")])))
	elif paired:
		return sorted([col for col in col_names if col.endswith("_paired")])
	elif difference:
		return sorted([col for col in col_names if col.endswith("_diff")])
	else:
		return sorted(list(col_names))


def add_difference_features(input_data):
	comparable_names = get_variable_names(input_data.columns.values, comparable=True, paired=False)
	for feat in comparable_names:
		input_data[feat+"_diff"] = input_data[feat+"_complex"].astype(np.float) - input_data[feat+"_simple"].astype(np.float)
	return input_data


def change_dtype(input_data, col_names, comparable=True):
	if comparable:
		old_names = col_names
		col_names = list()
		for col in old_names:
			col_names.append(col+"_complex")
			col_names.append(col+"_simple")
	# do_statistics.py:409: DtypeWarning: Columns (54,55,56,60,61,62) have mixed types. Specify dtype option on import or set low_memory=False.
	# en newsela 2015
	input_data.replace(False, 0, inplace=True)
	input_data.replace("False", 0, inplace=True)
	input_data.replace(True, 1, inplace=True)
	input_data.replace("True", 1, inplace=True)
	input_data[col_names] = input_data[col_names].apply(pd.to_numeric)
	return input_data


def test_distribution_null_hypothesis(complex_values, simple_values, independent, feat_name, dict_path="feature_dict_checked.json"):
	complex_values = complex_values[complex_values.notnull()]
	simple_values = simple_values[simple_values.notnull()]
	# todo: remove if all values 0 or nan
	if len(complex_values) == 0 or len(simple_values) == 0 or \
			(complex_values == 0).sum() == len(complex_values) or \
			(simple_values == 0).sum() == len(simple_values) or \
			list(complex_values) == list(simple_values):
		return ("0", 0, 0, None)
	# # 0: nominal, 1: ordinal, 2: interval, 3: ratio
	# scale_of_measurement = check_scale(complex_values)
	scale_of_measurement = check_scale_from_dict(dict_path, "comparable", feat_name)
	normal_distribution = check_distribution([complex_values, simple_values], p_threshold=0.05)
	variance_homogeneity = check_variance_homogeneity([complex_values, simple_values], p_threshold=0.05)
	if scale_of_measurement >= 2 and normal_distribution and variance_homogeneity and independent:
		t_value, p_value = stats.ttest_ind(complex_values, simple_values, equal_var=True)
		effect_size = abs(math.sqrt(t_value ** 2 / (t_value ** 2 + min(complex_values, simple_values) - 1)))
		return ("Student's t-test", t_value, p_value, effect_size)
	elif scale_of_measurement >= 2 and normal_distribution and not variance_homogeneity and independent:
		t_value, p_value = stats.ttest_ind(complex_values, simple_values, equal_var=False)
		effect_size = abs(math.sqrt(t_value ** 2 / (t_value ** 2 + min(complex_values, simple_values) - 1)))
		return ("Welch's t-test", t_value, p_value, effect_size)
	elif scale_of_measurement >= 1 and independent:
		t_value, p_value = stats.mannwhitneyu(complex_values, simple_values)
		#effect_size = get_effect_size(t_value, min(len(complex_values), len(simple_values)))
		return ("Mann–Whitney U test", t_value, p_value, None)
	elif scale_of_measurement >= 2 and normal_distribution and variance_homogeneity and not independent:
		t_value, p_value = stats.ttest_rel(complex_values, simple_values)
		# effect_size = abs(math.sqrt(t_value**2/(t_value**2+min(complex_values, simple_values)-1)))
		effect_size = stats.pearsonr(complex_values, simple_values)[0]
		return ("Student's t-test", t_value, p_value, effect_size)
	elif scale_of_measurement >= 1 and not independent:
		if len(complex_values) != len(simple_values):
			return ("No test", np.mean(complex_values), np.mean(simple_values), None)
		t_value, p_value, z_value = wilcoxon(complex_values, simple_values)
		effect_size = abs(z_value/math.sqrt(min(len(complex_values), len(simple_values))))
		#effect_size = stats.pearsonr(complex_values, simple_values)[0]
		return ("Wilcoxon signed-rank test", t_value, p_value, effect_size)
	else:
		# todo name only distribution of values?
		return ("No test", np.mean(complex_values), np.mean(simple_values), None)


def posthoc_dunn_z(a, val_col=None, group_col=None, p_adjust=None, sort=True):

    '''Post hoc pairwise test for multiple comparisons of mean rank sums
    (Dunn's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_, [2]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details. Available methods are:
        'bonferroni' : one-step correction
        'sidak' : one-step correction
        'holm-sidak' : step-down method using Sidak adjustments
        'holm' : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel' : closed method based on Simes tests (non-negative)
        'fdr_bh' : Benjamini/Hochberg  (non-negative)
        'fdr_by' : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (non-negative)
        'fdr_tsbky' : two stage fdr correction (non-negative)

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    Returns
    -------
    result : pandas DataFrame
        P values.

    Notes
    -----
    A tie correction will be employed according to Glantz (2012).

    References
    ----------
    .. [1] O.J. Dunn (1964). Multiple comparisons using rank sums.
        Technometrics, 6, 241-252.
    .. [2] S.A. Glantz (2012), Primer of Biostatistics. New York: McGraw Hill.

    Examples
    --------

    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_dunn(x, p_adjust = 'holm')
    '''

    def compare_dunn_z(i, j):
        diff = np.abs(x_ranks_avg.loc[i] - x_ranks_avg.loc[j])
        A = n * (n + 1.) / 12.
        B = (1. / x_lens.loc[i] + 1. / x_lens.loc[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        #p_value = 2. * ss.norm.sf(np.abs(z_value))
        return z_value

    x, _val_col, _group_col = scikit_posthocs.__convert_to_df(a, val_col, group_col)

    if not sort:
        x[_group_col] = pd.Categorical(x[_group_col], categories=x[_group_col].unique(), ordered=True)

    x.sort_values(by=[_group_col, _val_col], ascending=True, inplace=True)
    n = len(x.index)
    x_groups_unique = np.unique(x[_group_col])
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()

    # ties
    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = tie_sum / (12. * (n - 1))

    vs = np.zeros((x_len, x_len))
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:,:] = 0

    for i,j in combs:
        vs[i, j] = compare_dunn_z(x_groups_unique[i], x_groups_unique[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method = p_adjust)[1]

    vs[tri_lower] = vs.T[tri_lower]
    np.fill_diagonal(vs, -1)
    return pd.DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)


def compare_languages(list_lang_results, feat_name, list_corpus_names, p_threshold=0.05, dict_path="feature_dict_checked.json"):
	list_lang_no_nan = list()
	corpus_names = OrderedDict()
	for lang_values, corpus_name in zip(list_lang_results, list_corpus_names):
		no_nans = lang_values[lang_values.notnull()]
		if len(no_nans) > 0:
			list_lang_no_nan.append(no_nans)
			corpus_names[corpus_name] = len(no_nans)
	if len(list_lang_no_nan) == 0:
		return 0,0
	# scale_of_measurement = check_scale(list_lang_no_nan[0])
	scale_of_measurement = check_scale_from_dict(dict_path, "paired", feat_name)
	# # 0: nominal, 1: ordinal, 2: interval, 3: ratio
	normal_distribution = check_distribution(list_lang_no_nan, p_threshold=0.05)
	variance_homogeneity = check_variance_homogeneity(list_lang_no_nan, p_threshold=0.05)
	if scale_of_measurement >= 2 and normal_distribution and variance_homogeneity:
		# does the language affect the value of the feature? Does simplifications for each langauge work similar?
		t_value, p_value = stats.f_oneway(*list_lang_no_nan)
		return ("ANOVA", p_value)
		#if p_value <= p_threshold:
			# posthoc: which langauges are different?
			# stats.multicomp.pairwise_tukeyhsd
			# if two different ones found, use pearson to get effect size
			#effect_size = stats.pearsonr(complex_values, simple_values)[0]
			# effec_size = cohend(complex_values, simple_values)
	elif scale_of_measurement >= 1:
		try:
			h_statistic, p_value = stats.kruskal(*list_lang_no_nan)
		except ValueError:
			return 0,0
		if 0 < p_value <= p_threshold:
			if p_value <= 0.01:
				p_value = "p<=.01"
			elif p_value <= 0.05:
				p_value = "p<=.05"
			else:
				p_value = "p>0.05"
			output_list = list()
			posthoc_frame = scikit_posthocs.posthoc_dunn(list_lang_no_nan, p_adjust="holm")
			posthoc_frame_z = posthoc_dunn_z(list_lang_no_nan)
			for i, name_corpus_col in zip(posthoc_frame.columns.values, corpus_names.keys()):
				for n, name_corpus_row in zip(range(0, len(posthoc_frame)), corpus_names.keys()):
					if p_threshold >= posthoc_frame.iloc[n][i] > 0:
						effect_size = abs(posthoc_frame_z.iloc[n][i]/math.sqrt(corpus_names[name_corpus_col]+corpus_names[name_corpus_row]))
						if effect_size >= 0.1:
							output_list.append(["Kruskal ", p_value, "effectsize", str(round(effect_size, 4)),
												"h", str(round(h_statistic, 4)), "z", str(round(posthoc_frame_z.iloc[n][i],4)), name_corpus_col, name_corpus_row])
					#pos_col = list(corpus_names.keys()).index(name_corpus_col)
						#pos_row = list(corpus_names.keys()).index(name_corpus_row)
						#effect_size_pearson = stats.pearsonr(list_lang_no_nan[pos_col], list_lang_no_nan[pos_row])[0]
						# print(len(list_lang_no_nan[pos_col]), len(list_lang_no_nan[pos_row]))
						# effect_size_cohen = cohend(list_lang_no_nan[pos_col], list_lang_no_nan[pos_row])
			return output_list
		else:
			return 0, 0
	else:
		return 0, 0


def cohend(d1, d2):
	# code from here https://machinelearningmastery.com/effect-size-measures-in-python/
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s


def get_descriptive_values(input_values):
	input_values = input_values[input_values.notnull()]
	return len(input_values), np.mean(input_values), np.std(input_values)


def get_effect_size(z_value, n):
	return abs(z_value/math.sqrt(n))

def scale_value_to_text(value):
	dict_scale = {0: "nominal", 1: "ordinal", 2: "interval", 3: "ratio"}
	return dict_scale[value]


def check_scale(input_series):
	# 0: nominal, 1: ordinal, 2: interval, 3: ratio
	# enough to check one scale because both have equal values
	if len(set(input_series).difference({0,1})) == 0:  #input_series.all() in [0, 1]:
		return 0
	elif all(0 <= i <= 1 for i in input_series):
		return 3
	else:
		return 1
	# if len(values.difference({0,1})) <= 1:
	# 	# including nan value
	# 	return "nominal"
	# else:
	# 	return "interval"


def check_scale_from_dict(dict_path, comparable_or_paired, feat_name):
	with open(dict_path) as f:
		data = json.load(f)
	if feat_name in data[comparable_or_paired].keys():
		return data[comparable_or_paired][feat_name]["measurement_scale"]
	else:
		#print(feat_name, " no information in feature dict provided.")
		return 1


def check_distribution(list_series, p_threshold=0.05):
	normal_distribution = False
	for input_series in list_series:
		w, p_value = stats.shapiro(input_series)
		if p_value >= p_threshold:
			# if significant no normal distribution, hence p_value must be greater or equal to threshold
			normal_distribution = True
		else:
			normal_distribution = False
	return normal_distribution


def check_variance_homogeneity(list_values, p_threshold=0.05):
	w, p_value = stats.levene(*list_values)
	if p_value >= p_threshold:
		# if significant then the values are heterogeneous, hence p_value must be greater or equal to threshold
		return True
	else:
		return False


def strong_effect_bold(val):
	# bold = 'bold' if not isinstance(val, str) and float(val) >= 0.5 else ''
	# return 'font-weight: %s' % bold
	if isinstance(val, str):
		color = 'black'
	elif float(val) >= 0.4:
		color = "darkblue"
	elif float(val) >= 0.25:
		color = "darkgreen"
	else:
		color = "violet"
	return 'color: %s' % color


def get_effect_stars(p_val, effect_size, p_threshold=0.05):
	if p_val <= p_threshold:
		if effect_size >= 0.4:
			return "***"
		elif effect_size >= 0.25:
			return "**"
		elif effect_size >= 0.1:
			return "*"
		else:
			return ""
	else:
		return ""


def get_statistics(input_data, comparable_col_names, paired_col_names, corpus_name, output_file_text, output_file_descriptive_table, output_file_effect_table, p_threshold=0.05, key=""):
	result_sents = list()
	result_table = pd.DataFrame(columns=["feature", corpus_name])
	columns_descr = pd.MultiIndex.from_tuples(
		[("feature", ""), (corpus_name, "complex"), (corpus_name, "simple"),(corpus_name, "effect size")])
	#columns_descr = pd.MultiIndex.from_tuples([("feature", ""), (corpus_name, "N"), (corpus_name, "AVG (SD) complex"), (corpus_name, "AVG (SD) simple"), ("effect_size", "")])
		#[["feature", corpus_name], ["", "N", "AVG (SD) complex", "AVG (SD) simple"]])
	descriptive_table = pd.DataFrame(columns=columns_descr)
	columns_descr_paired = pd.MultiIndex.from_tuples([("feature", ""), (corpus_name, "N"), (corpus_name, "AVG paired"), (corpus_name, "SD paired")])
	descriptive_table_paired = pd.DataFrame(columns=columns_descr_paired)
	# print(input_data.describe())
	# print(comparable_col_names)
	for i, col in enumerate(comparable_col_names):
		#if col in ["check_if_head_is_noun", "check_if_head_is_verb",  "check_if_one_child_of_root_is_subject",  "check_passive_voice",
		#				"count_characters", "count_sentences", "count_syllables_in_sentence", "get_average_length_NP",
		#				"get_average_length_VP", "get_avg_length_PP", "get_ratio_named_entities",
		#				"get_ratio_of_interjections", "get_ratio_of_particles", "get_ratio_of_symbols",
		#				"get_ratio_referential", "is_non_projective"]:
		#	continue
		# print(col, corpus_name, len(input_data[input_data[col+"_complex"].notnull()]), len(input_data[input_data[col+"_simple"].notnull()]))
		test_name, t_value, p_value, effect_size = test_distribution_null_hypothesis(input_data[col+"_complex"], input_data[col+"_simple"], False, col)
		n_complex, avg_complex, sd_complex = get_descriptive_values(input_data[col+"_complex"])
		n_simple, avg_simple, sd_simple = get_descriptive_values(input_data[col + "_simple"])
		# print(col, test_name, t_value, p_value, effect_size, "complex", n_complex, avg_complex, sd_complex, "simple", n_simple, avg_simple, sd_simple)
		result_sent = get_result_sent(test_name, col, corpus_name, p_value, n_complex, avg_complex, sd_complex, n_simple, avg_simple, sd_simple, min(n_complex, n_simple)-1, t_value, effect_size, p_threshold=0.05, only_relevant=True)
		if result_sent:
			result_sents.append(result_sent)
		if effect_size == None:
			effect_size = 0
		if p_value > p_threshold or effect_size < 0.1:
			result_table.loc[i] = [col, ""]
		else:
			result_table.loc[i] = [col, str(round(effect_size,2))+get_p_value_stars(p_value)]
		descriptive_table.loc[i] = [col, str(round(avg_complex, 2))+"$\pm$"+str(round(sd_complex,2))+"", str(round(avg_simple, 2))+"$\pm$"+str(round(sd_simple,2))+"", get_effect_stars(p_value, effect_size, p_threshold=0.05)]
	descriptive_table.loc[i+1] = ["N", "", n_complex, ""]
	for n, col in enumerate(paired_col_names):
		n_paired, avg_paired, sd_paired = get_descriptive_values(input_data[col])
		# print(col, test_name, t_value, p_value, effect_size, "complex", n_complex, avg_complex, sd_complex, "simple", n_simple, avg_simple, sd_simple)
		descriptive_table_paired.loc[n] = [col, n_paired,
									round(avg_paired, 2), "$\pm$" + str(round(sd_paired, 2))]

	if output_file_text:
		with open(output_file_text, "w+") as f:
			f.writelines(result_sents)
	with open(output_file_effect_table, "w+") as f:
		f.write(result_table.to_latex(index=False, escape=False)+"\n\n")
	result_table.set_index("feature")
	# result_table_excel = result_table.style.applymap(strong_effect_bold)
	# result_table_excel.to_excel(corpus_name+'styled.xlsx', engine='openpyxl')
	# if output_file_table:
	with open(output_file_descriptive_table, "w+") as f:
		f.write(descriptive_table.to_latex(index=False, escape=False))
	return input_data, descriptive_table, result_table, descriptive_table_paired


def save_results(concat_descr, concat_effect, concat_descr_paired, output_descr_paired, type_value=""):
	type_value_dir = ""
	if type_value and not os.path.exists("data/results/"+type_value):
		os.makedirs("data/results/"+type_value)
		type_value_dir = type_value+"/"
	with open("data/results/"+type_value_dir+"all_descr_results"+type_value+".txt", "w") as f:
		f.write(concat_descr.to_latex(index=False, escape=False))
	with open("data/results/"+type_value_dir+"all_descr_results"+type_value+".csv", "w") as f:
		f.write(concat_descr.to_csv(index=False))

	with open("data/results/"+type_value_dir+"all_effect_results"+type_value+".txt", "w") as f:
		f.write(concat_effect.to_latex(index=False, escape=False))
	with open("data/results/"+type_value_dir+"all_effect_results"+type_value+".csv", "w") as f:
		f.write(concat_effect.to_csv(index=False))

	with open("data/results/"+type_value_dir+"all_descr_paired_results"+type_value+".txt", "w") as f:
		f.write(concat_descr_paired.to_latex(index=False, escape=False))
	with open("data/results/"+type_value_dir+"all_descr_paired_results.csv", "w") as f:
		f.write(concat_descr_paired.to_csv(index=False))

	with open("data/results/"+type_value_dir+"all_effect_paired_results"+type_value+".txt", "w") as f:
		f.write(output_descr_paired)
	return 1


def get_feature_dict(result_files):
	list_lang_input = list()
	for input_file in result_files:
		input_data = pd.read_csv("data/ALL/"+input_file, sep="\t", header=0, warn_bad_lines=True, error_bad_lines=False)
		# input_data = add_difference_features(input_data)
		list_lang_input.append(input_data)
	feature_dict = {"paired": {}, "comparable": {}}
	for input_data in list_lang_input:
		for feat in get_variable_names(input_data.columns.values, paired=True, comparable=False):
			if feat not in feature_dict["paired"].keys():
				feature_dict["paired"][feat] = {"description": "", "measurement_scale": check_scale(input_data[feat]),
												"measurement_scale_text": scale_value_to_text(check_scale(input_data[feat])),
												"min": min(input_data[feat]), "max": max(input_data[feat]),
												"type": ""}
			else:
				if min(input_data[feat]) < feature_dict["paired"][feat]["min"]:
					feature_dict["paired"][feat]["min"] = min(input_data[feat])
				if max(input_data[feat]) > feature_dict["paired"][feat]["max"]:
					feature_dict["paired"][feat]["max"] = max(input_data[feat])
				if feature_dict["paired"][feat]["measurement_scale"] < check_scale(input_data[feat]) < 3:
					feature_dict["paired"][feat]["measurement_scale"] = check_scale(input_data[feat])
					feature_dict["paired"][feat]["measurement_scale_text"] = scale_value_to_text(feature_dict["paired"][feat]["measurement_scale"])
		for feat in get_variable_names(input_data.columns.values, paired=False, comparable=True):
			if feat not in feature_dict["comparable"].keys():
				feature_dict["comparable"][feat] = {"description": "",
													"measurement_scale_text": scale_value_to_text(max(check_scale(input_data[feat + "_complex"]), check_scale(input_data[feat + "_simple"]))),
													"measurement_scale": max(check_scale(input_data[feat + "_complex"]), check_scale(input_data[feat + "_simple"])),
													"min": min(min(input_data[feat+"_complex"]), min(input_data[feat+"_simple"])),
													"max": max(max(input_data[feat+"_complex"]), max(input_data[feat+"_simple"])),
													"type": ""}
			else:
				if min(input_data[feat+"_complex"]) < feature_dict["comparable"][feat]["min"]:
					feature_dict["comparable"][feat]["min"] = min(input_data[feat+"_complex"])
				if min(input_data[feat+"_simple"]) < feature_dict["comparable"][feat]["min"]:
					feature_dict["comparable"][feat]["min"] = min(input_data[feat+"_simple"])
				if max(input_data[feat+"_complex"]) > feature_dict["comparable"][feat]["max"]:
					feature_dict["comparable"][feat]["max"] = max(input_data[feat+"_complex"])
				if max(input_data[feat+"_simple"]) > feature_dict["comparable"][feat]["max"]:
					feature_dict["comparable"][feat]["max"] = max(input_data[feat+"_simple"])
				if feature_dict["comparable"][feat]["measurement_scale"] < max(check_scale(input_data[feat + "_complex"]), check_scale(input_data[feat + "_simple"])) < 3:
					feature_dict["comparable"][feat]["measurement_scale"] = max(check_scale(input_data[feat + "_complex"]), check_scale(input_data[feat + "_simple"]))
					feature_dict["comparable"][feat]["measurement_scale_text"] = scale_value_to_text(feature_dict["comparable"][feat]["measurement_scale"])

	with open("feature_dict.json", "w+") as f:
		json.dump(feature_dict, f, sort_keys=True, indent=4)
	return feature_dict


def preprocess_input_data(input_file):
	input_data = pd.read_csv("data/ALL/"+input_file, sep="\t", header=0, warn_bad_lines=True, error_bad_lines=False,
							 quoting=csv.QUOTE_NONE, encoding='utf-8')
	comparable_col_names = get_variable_names(input_data.columns.values, comparable=True, paired=False)
	input_data = add_difference_features(input_data)
	paired_col_names = get_variable_names(input_data.columns.values, paired=True, comparable=False)
	paired_col_names = paired_col_names + get_variable_names(input_data.columns.values, paired=False, comparable=False,
															 difference=True)

	input_data = change_dtype(input_data, comparable_col_names, comparable=True)
	input_data = change_dtype(input_data, paired_col_names, comparable=False)
	return input_data, comparable_col_names, paired_col_names


def stack_corpora(results_files):
	stacked_data, comparable_col_names, paired_col_names, corpus_name = "", "", "", ""
	for f, input_file in enumerate(results_files):
		corpus_name = input_file.split("/")[-1][:-4]
		input_data, comparable_col_names, paired_col_names = preprocess_input_data(input_file)
		if f == 0:
			stacked_data = input_data
		else:
			stacked_data = pd.concat([stacked_data, input_data])
	return stacked_data, comparable_col_names, paired_col_names, corpus_name


def get_statistics_for_stacked_domains(file_dict):
	for key in file_dict.keys():
		get_statistics_for_stacked_corpora(file_dict[key], key)

	return 1


def get_paired_statistics_for_crossdata(file_dict, cross_type="domain"):
	list_lang_input, corpus_names = list(), list()
	for f, corpus_name in enumerate(file_dict.keys()):
		print(corpus_name)
		stacked_data, comparable_col_names, paired_col_names, corpus_name = stack_corpora(file_dict[corpus_name])
		stacked_data, corpus_descr, corpus_effect, corpus_descr_paired = get_statistics(stacked_data, comparable_col_names,
													paired_col_names, corpus_name,
													 "data/results/" + corpus_name + "_sent_results.txt",
													 "data/results/" + corpus_name + "_descr_results.txt",
													 "data/results/" + corpus_name + "_effect_results.txt")
		list_lang_input.append(stacked_data)
		corpus_names.append(corpus_name)
		if f == 0:
			concat_descr = corpus_descr
			concat_effect = corpus_effect
			concat_descr_paired = corpus_descr_paired
		else:
			corpus_effect = corpus_effect.drop(['feature'], axis=1)
			corpus_descr = corpus_descr.drop('feature', axis=1, level=0)
			corpus_descr_paired = corpus_descr_paired.drop('feature', axis=1, level=0)
			concat_descr = pd.concat([concat_descr, corpus_descr], axis=1)
			concat_effect = pd.concat([concat_effect, corpus_effect], axis=1)
			concat_descr_paired = pd.concat([concat_descr_paired, corpus_descr_paired], axis=1)


	paired_col_names = get_variable_names(list_lang_input[0].columns.values, paired=True, comparable=False)
	paired_col_names = paired_col_names + get_variable_names(list_lang_input[0].columns.values, paired=False, comparable=False, difference=True)
	output_descr_paired = ""
	for col in paired_col_names:
		# print(col, len([lang_input[col] for lang_input in list_lang_input]))
		result = compare_languages([lang_input[col] for lang_input in list_lang_input], feat_name=col, list_corpus_names=corpus_names, p_threshold=0.05)
		if type(result) == list:
			for res in result:
				output_descr_paired += col + " " + " ".join(res)+ "\n"
		elif result[1] <= 0.05 and result[1] > 0.0:
			output_descr_paired += col+" " + result[0] + " " + str(result[1]) + "\n"
	save_results(concat_descr, concat_effect, concat_descr_paired, output_descr_paired, type_value=cross_type)
	return 1



def get_statistics_for_stacked_corpora(results_files, key="stacked_corpora"):
	"""for f, input_file in enumerate(results_files):
		corpus_name = input_file.split("/")[-1][:-4]
		input_data, comparable_col_names, paired_col_names = preprocess_input_data(input_file)
		if f == 0:
			stacked_data = input_data
		else:
			stacked_data = pd.concat([stacked_data, input_data])"""
	stacked_data, comparable_col_names, paired_col_names, corpus_name = stack_corpora(results_files)
	key_dir = ""
	if key:
		key_dir = key+"/"
		if not os.path.exists("data/results/"+key):
			os.makedirs("data/results/"+key)
	input_data, corpus_descr, corpus_effect, corpus_descr_paired = get_statistics(stacked_data, comparable_col_names,
								paired_col_names, corpus_name,
								 "data/results/"+key_dir+"sent_results_stacked_"+key+".txt",
								 "data/results/"+key_dir+"descr_results_stacked_"+key+".txt",
								 "data/results/"+key_dir+"effect_results_stacked_"+key+".txt")
	return input_data, corpus_descr, corpus_effect, corpus_descr_paired


def get_statistics_for_all_corpora(result_files, type_value=""):
	list_lang_input = list()
	corpus_names = list()
	type_value_dir = ""
	if type_value:
		type_value_dir = type_value+"/"
		if not os.path.exists("data/results/"+type_value):
			os.makedirs("data/results/"+type_value)
	for f, input_file in enumerate(result_files):
		corpus_name = input_file.split("/")[-1][:-4]
		print(input_file)
		input_data, comparable_col_names, paired_col_names = preprocess_input_data(input_file)
		input_data, corpus_descr, corpus_effect, corpus_descr_paired = get_statistics(input_data, comparable_col_names,
													paired_col_names, corpus_name,
													 "data/results/"+ type_value_dir + corpus_name + "_sent_results.txt",
													 "data/results/" + type_value_dir + corpus_name + "_descr_results.txt",
													 "data/results/" + type_value_dir + corpus_name + "_effect_results.txt")
		list_lang_input.append(input_data)
		corpus_names.append(corpus_name)
		if f == 0:
			concat_descr = corpus_descr
			concat_effect = corpus_effect
			concat_descr_paired = corpus_descr_paired
		else:
			corpus_effect = corpus_effect.drop(['feature'], axis=1)
			corpus_descr = corpus_descr.drop('feature', axis=1, level=0)
			corpus_descr_paired = corpus_descr_paired.drop('feature', axis=1, level=0)
			concat_descr = pd.concat([concat_descr, corpus_descr], axis=1)
			concat_effect = pd.concat([concat_effect, corpus_effect], axis=1)
			concat_descr_paired = pd.concat([concat_descr_paired, corpus_descr_paired], axis=1)

	paired_col_names = get_variable_names(list_lang_input[0].columns.values, paired=True, comparable=False)
	paired_col_names = paired_col_names + get_variable_names(list_lang_input[0].columns.values, paired=False, comparable=False, difference=True)
	output_descr_paired = ""
	for col in paired_col_names:
		# print(col, len([lang_input[col] for lang_input in list_lang_input]))
		result = compare_languages([lang_input[col] for lang_input in list_lang_input], feat_name=col, list_corpus_names=corpus_names, p_threshold=0.05)
		if type(result) == list:
			for res in result:
				output_descr_paired += col + " " + " ".join(res)+ "\n"
		elif result[1] <= 0.05 and result[1] > 0.0:
			output_descr_paired += col+" " + result[0] + " " + str(result[1]) + "\n"
	save_results(concat_descr, concat_effect, concat_descr_paired, output_descr_paired, type_value=type_value)
	return 1


def logistic_regression_model(result_files, output_name, complete=False):
	r2_value = 0
	if complete:
		output_frame = pd.DataFrame()
	for f, input_file in enumerate(result_files):
		corpus_name = input_file.split("/")[-1][:-4]
		print(corpus_name)
		input_data = pd.read_csv("data/ALL/"+corpus_name+".tsv", sep="\t", header=0, warn_bad_lines=True, error_bad_lines=False,
								 quoting=csv.QUOTE_NONE, encoding='utf-8')
		comparable_names = get_variable_names(input_data.columns.values, comparable=True, paired=False)
		input_data = change_dtype(input_data, comparable_names, comparable=True)
		nan_cols = set(["_".join(col.split("_")[:-1]) for col in input_data.columns[input_data.isnull().any()]])
		comparable_names = sorted(list(set(comparable_names).difference(nan_cols)))

		columns_complex = [col+'_complex' for col in comparable_names]
		columns_simple = [col+'_simple' for col in comparable_names]
		complex_values = input_data[columns_complex]
		complex_values.columns = comparable_names
		simple_values = input_data[columns_simple]
		simple_values.columns = comparable_names
		label_complex = [0]*len(input_data[columns_complex])
		label_simple = [1]*len(input_data[columns_simple])
		X_train, X_test, y_train, y_test = train_test_split(pd.concat([complex_values, simple_values], sort=True), label_complex+label_simple)
		logmodel = LogisticRegression(C=1.0, multi_class="ovr", max_iter=10000)
		if complete:
			clf = logmodel.fit(pd.concat([X_train, X_test], sort=True), y_train+y_test)
		else:
			clf = logmodel.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			r2_value = r2_score(y_test, y_pred)
		coefs = logmodel.coef_[0]
		coefs_values = sorted([(name, value) for name, value in zip(comparable_names, coefs)], key=lambda tup: tup[1], reverse=True)


		output_path = 'data/results/'+output_name+"/"+corpus_name+'_logReg.csv'
		if complete:
			if f == 0:
				output_frame_all = pd.DataFrame()
				output_frame_all["feature_name"] = [pair[0] for pair in coefs_values]
				output_frame_all = output_frame_all.set_index("feature_name")
				output_frame_all[corpus_name] = [round(pair[1], 4) for pair in coefs_values]
				output_frame_all[corpus_name+"_rank"] = range(0, len(coefs_values))
			else:
				output_frame = pd.DataFrame()
				output_frame["feature_name"] = [pair[0] for pair in coefs_values]
				output_frame = output_frame.set_index("feature_name")
				output_frame[corpus_name] = [round(pair[1], 4) for pair in coefs_values]
				output_frame[corpus_name+"_rank"] = range(0, len(coefs_values))
				output_frame_all = pd.concat([output_frame_all, output_frame], sort=True, axis=1)
		with open(output_path, 'w+') as out:
			csv_out = csv.writer(out)
			csv_out.writerow(['feature_name', 'value'])
			for row in coefs_values:
				csv_out.writerow(row)
			csv_out.writerow(["R2 "+str(round(r2_value, 4)), "Effectsize "+ str(round(math.sqrt(r2_value/1-r2_value),4))])
	if complete:
		output_path = 'data/results/' + output_name + 'all_logReg_full.csv'
		output_frame_all.to_csv(output_path)
	return 1


def average_of_log(input_path, output_path, std=False):
	log_dirs = [directory for directory in os.listdir(input_path) if directory.startswith("log") and os.path.isdir(input_path+"/"+directory)]
	corpora = sorted([corpus for corpus in os.listdir(input_path+"/"+log_dirs[0])])
	feature_names = set()
	output_dict = dict()
	output_dict["n"] = len(log_dirs)
	for i, corpus in enumerate(corpora):
		corpus_name = corpus.split("_")[0]
		for n, log_dir in enumerate(log_dirs):
			if n == 0:
				corpus_frame = pd.read_csv(input_path+"/"+log_dir+"/"+corpus, header=0)
				corpus_frame = corpus_frame.set_index('feature_name').transpose()
			else:
				corpus_log_frame = pd.read_csv(input_path + "/" + log_dir + "/" + corpus, header=0)
				corpus_log_frame = corpus_log_frame.set_index("feature_name").transpose()
				corpus_frame = pd.concat([corpus_frame, corpus_log_frame], axis=0, sort=True)
				feature_names.union(set(corpus_frame.columns.values))
		output_dict[corpus_name+"_avg"] = {}
		output_dict[corpus_name + "_rank"] = {}
		if std:
			output_dict[corpus_name + "_std"] = {}
		for col in corpus_frame.columns.values:
			output_dict[corpus_name+"_avg"][col] = round(corpus_frame[col].mean(),4)
			if std:
				output_dict[corpus_name+"_std"][col] = round(corpus_frame[col].std(),4)
		order_feats = [k for k, v in
					   sorted(output_dict[corpus_name + "_avg"].items(), key=lambda item: item[1], reverse=True)]
		for col in corpus_frame.columns.values:
			output_dict[corpus_name + "_rank"][col] = order_feats.index(col)

	output_frame = pd.DataFrame(output_dict)
	output_frame.to_csv("data/results/all_logReg_avg.csv")


def main(argv):
	result_files = sorted(list(os.listdir("data/ALL")))
	# result_files = [file for file in result_files if not "en-Newsela_2016" in file and not "es-Newsela" in file]
	print(result_files)
	if not os.path.exists("data/results"):
		os.mkdir("data/results")
	# get_feature_dict(result_files)
	print("do statistics for all corpora solely")
	get_statistics_for_all_corpora(result_files, "all")
	# logistic_regression_model(result_files, "", complete=True)
	file_dict = {
		"news": ["cs-COSTRA.tsv", "es-Newsela45.tsv", "es-Newsela01.tsv", "en-Newsela_2015.tsv", "en-Newsela_201601.tsv"],
		 "wiki": ["en-TurkCorpus.tsv", "en-QATS.tsv"],
		 "web": ["de-Klaper.tsv", "it-PaCCSS.tsv"],
		 "NewselaEN": ["en-Newsela_201601.tsv", "en-Newsela_201612.tsv", "en-Newsela_201623.tsv", "en-Newsela_201634.tsv", "en-Newsela_201645.tsv"],
		 "NewselaES": ["es-Newsela01.tsv", "es-Newsela12.tsv", "es-Newsela23.tsv", "es-Newsela34.tsv", "es-Newsela45.tsv"],
		 "EN": ["en-TurkCorpus.tsv", "en-Newsela_2015.tsv", "en-Newsela_201601.tsv", "en-QATS.tsv"]
		}
	dict_domain = {
		"news": ["cs-COSTRA.tsv", "es-Newsela45.tsv", "en-Newsela_2015.tsv", "en-Newsela_201601.tsv"  "es-Newsela01.tsv"],
		"wiki": ["en-TurkCorpus.tsv", "en-QATS.tsv"],
		 "web": ["de-Klaper.tsv", "it-PaCCSS.tsv"]
		}
	dict_lang = {
		"EN": ["en-TurkCorpus.tsv", "en-Newsela_2015.tsv", "en-Newsela_201601.tsv", "en-QATS.tsv"],
		"DE": ['de-Klaper.tsv'],
		"CS": ['cs-COSTRA.tsv'],
		"ES": ['es-Newsela45.tsv', 'es-Newsela01.tsv', 'es-Newsela12.tsv', 'es-Newsela23.tsv', 'es-Newsela34.tsv'],
		"IT": ['it-PaCCSS.tsv']
		}

	print("do statistics per domain and language")
	for key in file_dict.keys():
		get_statistics_for_all_corpora(file_dict[key], key)
	print("do statistics for stacked languages")
	get_statistics_for_stacked_corpora(result_files)
	print("do statistics for stacked domains")
	get_statistics_for_stacked_domains(file_dict)
	print("do statistics per domain cross data")
	get_paired_statistics_for_crossdata(dict_domain, "cross_domain")
	print("do statistics per language cross data")
	get_paired_statistics_for_crossdata(dict_lang, "cross-lingual")

	# for i in range(0, 6):  # range(0,11)
	# 	print(i)
	# 	if not os.path.exists("data/results/log_"+str(i)):
	# 		os.mkdir("data/results/log_"+str(i))
	# 	logistic_regression_model(result_files, "log_"+str(i)+'/', complete=False)
		# test how good regression predicts labels using features? or use complete data set (train= train+test)?
	# logistic_regression_model(result_files, "", complete=True)
	# average_of_log("data/results/", "", std=False)




if __name__ == "__main__":
	main(sys.argv[1:])
