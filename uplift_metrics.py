"""
Metrics for uplift-modeling.

Currently contains:
-expected_conversion_rate(): expected conversion rate given treatment plan (scoring) and k (number
of samples to be treated in the dataset)
-expected uplift(): the difference in conversion rates of two treatment plans (e.g.
 scoring and random)
-auuc_metric(): expected uplift some given treatment plan (scoring) and no prior
 preference on treatment/control split (a.k.a AUUC following Jaskowski &
 Jarosewicz 2012).
-plot_conversion_rates(): plots the conversion rate as function of fraction of
samples treated.
-qini_coefficient(): estimates the qini-coefficient using Radcliffe's formulas

Could be added:
-conversion_r(): a function essentially estimating what k
 corresponds to a given r (and calling expected_conversion_rate)
-conversion_s(): a function estimating what k corresponds
 to a given score threshold (and calling expected_conversion_rate)
-conversion_for_fixed_plan(): instead of taking scoring
 and k, just using a binary vector for estimation.
-uplift_by_gross() implementing the special case defined by Gross & Tibshirani (2016?)
-AUL() as defined by Diemert & al (i.e. M - rand?)
-AUUC() as defined by Diemert & al.


To Do:
-Decide what to do about AUUC and AUL. Compare to other metrics as well.
AUUC seems to be an unnormalized version of the qini coefficient where
group imbalance is not normalized and the resulting value is not divided
by some "optimal" score.

Known issues:
-There is still the issue with estimating conversion rate from zero samples in
 expected_conversion_rate() for small and large k. Currently we have set it to
 zero to match the approach by Diemert & al. but it is not particularly
 "correct."
"""


import csv
from datetime import datetime
import warnings
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import beta


class UpliftMetrics():
    """Class for a collection of metrics relating to uplift modeling.
    The main purpose of this class is to better keep track of metrics
    relating to tests and to automate parts of it. Initialization
    causes estimation of all metrics and self.save_to_csv() stores
    the metrics.

    Attributes:
    algorithm (str): type of algorithm, e.g. 'double-classifier with lr'
    dataset (str): dataset used to obtain model and metrics
    THIS IS STILL WORK IN PROGRESS!

    Methods:
    __init__: Self-explanatory
    print_: Prints to screen in a predefined format
    write_to_csv: saves results with metadata to some predefined file
    """

    def __init__(self, data_class, data_prob, data_group,
                 test_name=None, test_description=None,
                 algorithm=None,
                 dataset=None,
                 parameters=None,
                 euce_k=100):
        """This method initiates a UpliftMetric object and estimates
        a number of useful metrics given the data.

        Args:
        data_class (np.array([bool])): Array of class labels for samples.
        data_prob (np.array([float])): Array of uplift predictions as
         probabilities for samples. Can also be replaced with data_score,
         but metrics relying on probabilitise will be way off.
        data_group (np.array([bool])): Array of group labels for samples.
        True indicates that sample belongs to the treatment group.
        euce_k (int): k to use for estimation of expected uplift calibration
         error.

        Attributes:
        e_r_conversion_rate: Expected uplift given no prior preference on
         treatment rate.
        auuc: Area under the uplift curve following Jaskowski & Jarosewicz
         2012. This is the expected value of conversion rate given no prior
         preference on fraction of samples to be treated (i.e. it is an integral
         over the treatment rate). Note that Jaskowski & Jarosewicz were
         not talking about expected values, but their formulas result in one.
         auuc = e_r_conversion_rate - e_r_conversion_rate_random
         Note that e_r_conversion_rate_random == e_conversion_rate_random
        improvement_to_random (float): e_r_conversion_rate_random /
         e_r_conversion_rate_random
        qini_coefficient (float): Qini coefficient as defined by Radcliffe 2007.
         The metric in question is outdated and has some theorethical flaws. We
         Have implemented tie handling (which Radcliffe says nothing about).
        ece_k (int): 'k' used to estimate expected calibration error and
         maximum calibration error.
        euce (float): Expected uplift calibration error. An extension of the
         expected calibration error defined for response modeling and suggested
         by Naeini & al. (2014).
        muce (float): Maximum uplit calibration error. An extension of the
         maximu calibration error defined for response modeling and suggested
         by Naeini & al. (2014).
        unique_scores (int): Number of unique uplift scores. Sanity check.
        samples (int): Number of samples used to estimate these metrics.

        Notes:
        e_r_* refers to Expected value of * _given_ no preference on fraction
         of samples to be treated.
        In analyzing the results so far, we have not been interested in how many
         samples have a positive uplift prediction vs. how many samples _should_
         be treated to reach a maximal conversion rate. Hence I will not store
         these for now.
        These metrics should be estimated once for every test, i.e. there will be
         some time to do this. The functions provided here will be used even
         though there could in principle be shortcuts.

        -Should there be a function for "quick-metrics"? I.e. metrics that are needed
        e.g. during training of neural networks? Skip for now.
        """
        self.algorithm = algorithm
        self.dataset = dataset
        self.parameters = parameters
        self.test_name = test_name
        self.test_description = test_description

        # Make a bunch of these based on the conversion_rates list.

        # Sanity check:
        if len(data_class) != len(data_prob) != len(data_group):
            raise Exception("Class, probability, and group vectors " +\
                            "needs to be of equal length!")

        # Sort dataset once (largest uplift first)
        data_idx = np.argsort(data_prob)[::-1]
        data_class = data_class[data_idx]
        data_prob = data_prob[data_idx]
        data_group = data_group[data_idx]

        # Calculate conversion-rate list
        tmp_1 = _expected_conversion_rates(data_class,
                                           data_prob,
                                           data_group)
        # Estimate metrics based on this:
        self.e_r_conversion_rate = np.mean(tmp_1)
        self.conversion_rate_no_treatments = tmp_1[0]
        self.conversion_rate_all_treatments = tmp_1[-1]
        self.e_conversion_rate_random = np.mean([self.conversion_rate_no_treatments,
                                                 self.conversion_rate_all_treatments])
        self.auuc = self.e_r_conversion_rate -\
                    self.e_conversion_rate_random * 1
        # Relative improvement to random.
        if self.e_conversion_rate_random != 0:
            self.improvement_to_random = self.auuc /\
                                         self.e_conversion_rate_random
        self.qini_coefficient = qini_coefficient(data_class, data_prob, data_group)
        self.euce_k = euce_k  # What's a good default value?
        tmp_2 = expected_uplift_calibration_error(data_class, data_prob, data_group,
                                                  k=self.euce_k)
        self.euce = tmp_2[0]  # Extract EUCE from tuple
        self.muce = tmp_2[1]  # Extract MUCE from tuple
        self.unique_scores = len(np.unique(data_prob))
        self.samples = len(data_prob)

    def __str__(self):
        """Function for e.g. easy printing of results to screen.
        """
        txt = "-" * 40 + "\n" +\
              "Test name: {0.test_name}\n".format(self) +\
              "Algorithm: {0.algorithm}\n".format(self) +\
              "dataset: {0.dataset} \n".format(self) +\
              "Test description: {0.test_description}\n".format(self) +\
              "E_r(conversion rate) \tAUUC \t\t\tEUCE \t\tMUCE \t\t" +\
              "Improvement to random [%] \tQini-coefficient \n" +\
              "{0.e_r_conversion_rate:9.9} \t\t{0.auuc:9.9} \t".format(self) +\
              "{0.euce:9.9} \t{0.muce:9.9} \t".format(self) +\
              "{0.improvement_to_random:9.9} \t\t\t".format(self) +\
              "{0.qini_coefficient:9.9} \n".format(self) +\
              "-" * 40
        return txt

    def write_to_csv(self, file_='uplift_results.csv'):
        """Function for storing metrics to csv-file in predefined format.
        The function will by default store the results after all other results,
        unless the file does not exist whereas it creates that file first.

        Notes:
        Python csv-library handles what could potentially break the format,
        e.g. strings like '";"'.
        """
        # 1. Check if file exists. If it does, append results. Otherwise
        # create first and add header row!
        write_new_headers = True
        if Path('./' + file_).exists():
            write_new_headers = False

        with open(file_, 'a') as resultfile:
            result_writer = csv.writer(resultfile, delimiter=';', quotechar='"')
            if write_new_headers:
                # Write new headers to file
                headers = ['Test name', 'Dataset', 'Test description',
                           'Algorithm', 'Parameters',
                           'Timestamp', 'E_r(conversion rate)',
                           'AUUC', 'Improvement to random [%]', 'Qini-coefficient',
                           'EUCE', 'MUCE', 'EUCE-k',
                           '#Unique scores', '#Samples',
                           'E(converison rate|No treatments)',
                           'E(conversion rate|All treatments)',
                           'E(conversion rate|random)']
                try:
                    result_writer.writerow(headers)
                except csv.Error:
                    print("Error in saving headers")

            # Include *everything* in result list:
            result_list = [self.test_name, self.dataset, self.test_description,
                           self.algorithm, self.parameters, str(datetime.utcnow()),
                           self.e_r_conversion_rate, self.auuc,
                           self.improvement_to_random * 100, self.qini_coefficient,
                           self.euce, self.muce, self.euce_k,
                           self.unique_scores, self.samples,
                           self.conversion_rate_no_treatments,
                           self.conversion_rate_all_treatments,
                           self.e_conversion_rate_random]
            try:
                result_writer.writerow(result_list)
            except csv.Error:
                print("Error in saving results to CSV.")
                print(result_list)


@jit(nopython=True)
def expected_conversion_rate(data_class,
                             data_score,
                             data_group,
                             k,
                             smoothing=0):
    """ Function for estimating expected conversion rate if we
    treated k/N fraction of all samples.

    Args:
    data_class (numpy.array, boolean): An array of labels for all samples
    data_score (numpy.array, float): An array of scores for every sample
    data_group (numpy.array, boolean): An array of labels for all samples.
    True indicates that the corresponding sample belongs to the treatment
    group, false indicates that it belongs to the control group.
    k (int): Number of samples that should be treated according to the
    treatment plan. k highest scoring samples are then treated.
    smoothing (float): Setting smoothing to something else than 0 enables
    smoothing when estimating the conversion rate. E.g. setting it to one
    corresponds to Laplace-smoothing.

    Implementatdion details:
    If k/N splits a clump of equally scoring samples, they are all
    treated as the "average" of this clump, i.e. the resulting conversion
    rate is an actual expected value.

    This function uses smoothing = 0 per default. This results in estimating
    the conversion rate of zero samples to 0. This happens frequently when
    we set k to something small or something very close to N (the total
    number of samples). This could become a problem if also N is small.

    Future ideas:
    Another option to smoothing could be to use a bayesian prior and
    perhaps estimate the expected value instead of maximum a posteriori
    or maximum likelihood.
    """

    if k == 0:
        # handle case where there are no samples in treatment group
        # i.e. where the conversion rate is estimated only from one group.
        # data_group == True, i.e. it is a treatment sample.
        control_conversions = np.sum(data_class[~data_group])
        control_samples = np.sum(~data_group)
        # There are no treated samples if k=0:
        conversion_rate = (control_conversions + smoothing) /\
                          (control_samples + 2 * smoothing)
    elif k == len(data_class):
        # handle case where there are no samples in control group
        treatment_conversions = np.sum(data_class[data_group])
        treatment_samples = np.sum(data_group)
        # All samples are treated:
        conversion_rate = (treatment_conversions + smoothing) /\
                          (treatment_samples + 2 * smoothing)
    else:
        # This is the "ordinary" flow.
        # Sort samples by decreasing score, i.e. the ones that should be treated first
        # are first:
        data_idx = np.argsort(data_score)[::-1]
        data_class = data_class[data_idx]
        data_score = data_score[data_idx]
        data_group = data_group[data_idx]

        # Handle case where k does not happen to comprise a border between two classes.
        # Three types of interesting samples: treatment samples with score < score[k],
        # control samples with score > score[k], and all samples with score == score[k].
        tot_samples = len(data_group)
        treatment_conversions = np.sum(data_class[(data_score > data_score[k - 1]) * data_group])
        treatment_samples = np.sum((data_score > data_score[k - 1]) * data_group)
        control_conversions = np.sum(data_class[(data_score < data_score[k - 1]) * ~data_group])
        control_samples = np.sum((data_score < data_score[k - 1]) * ~data_group)
        # Now we still need to count the samples where the score equals data_score[k]
        # This qpproach would remove the need for a sort.
        subset_group = data_group[data_score == data_score[k - 1]]
        subset_class = data_class[data_score == data_score[k - 1]]
        treatment_samples_in_subset = np.sum(subset_group)
        control_samples_in_subset = np.sum(~subset_group)
        samples_in_subset = len(subset_group)
        assert samples_in_subset == treatment_samples_in_subset + control_samples_in_subset,\
            "Mismatch in counting samples in subset?!?"
        treatment_conversions_in_subset = np.sum(subset_class[subset_group])
        control_conversions_in_subset = np.sum(subset_class[~subset_group])
        j = k - np.sum(data_score > data_score[k - 1])  # Split in subset corresponding to k
        treatment_conversions += j * treatment_conversions_in_subset /\
                                 samples_in_subset
        # Again, every sample in the subset with equal scores should be
        # treated as the "average sample" in the group!
        treatment_samples += j * np.sum(subset_group) / samples_in_subset
        control_conversions += (samples_in_subset - j) * control_conversions_in_subset /\
                               samples_in_subset
        control_samples += (samples_in_subset - j) * np.sum(~subset_group) / samples_in_subset
        treatment_conversion_rate = (treatment_conversions + smoothing) /\
                                    max(treatment_samples + 2 * smoothing, 1)
        control_conversion_rate = (control_conversions + smoothing) /\
                                  max(control_samples + 2 * smoothing, 1)
        conversion_rate = k / tot_samples * treatment_conversion_rate +\
                          (tot_samples - k) / tot_samples * control_conversion_rate

    return conversion_rate


def expected_uplift(data_class, data_score, data_group, k=None,
                    ref_score=None, ref_k=None,
                    ref_plan_type=None,
                    smoothing=0):
    """Function for estimating expected uplift for a given treatment
    plan w.r.t to a given reference plan. The treatment plan is
    defined by data_score and k, i.e. the k highest scoring samples
    are treated. This is a point estimate.
    With ref_plan_type = 'data', this is the formula presented by
    by Gross & Tibshirani in 2016.

    Args:
    data_class (np.array([bool])): Array of labels for the samples.
     True indicates positive label.
    data_score (np.array([float])): Array of uplift scores for the
     samples. Highest scoring samples are treated first.
    data_group (np.array([bool])): Array of group for the samples.
     True indicates that the sample is a treatment sample.
    ref_plan_type  in {'rand', 'data', 'comp', 'no_treatments',
     'all_treatments', 'plan_b'}
    """

    if k is None:
        k = sum(data_score > 0)
    # Expected conversion rate for treatment plan with k treated samples:
    conversion_for_plan = expected_conversion_rate(data_class, data_score, data_group, k, smoothing)
    if ref_plan_type == 'rand':
        raise Exception("Random treatment plan is currently not implemented")
        #  conversion_for_ref = conversion_rand(data_class, data_score, data_group, k, smoothing)
    elif ref_plan_type == 'data':
        # Use treatments as used in the data for reference.
        # This was used by Gross & Tibshirani (2016) with smoothing = 0.
        conversion_for_ref = (sum(data_class) + smoothing) / (len(data_class) + 2 * smoothing)
    elif ref_plan_type == 'comp':
        # Use some composite treatment plan for reference
        raise Exception("Composite reference plan is not implemented yet.")
    elif ref_plan_type == 'no_treatments':
        conversion_for_ref = expected_conversion_rate(data_class, data_score, data_group, 0, smoothing)
    elif ref_plan_type == 'all_treatments':
        conversion_for_ref = expected_conversion_rate(data_class, data_score, data_group, \
                                          len(data_group), smoothing)
    elif ref_plan_type == 'plan_b':
        # "plan_b" another scoring plan with ref_k
        conversion_for_ref = expected_conversion_rate(data_class, ref_score, data_group, ref_k, smoothing)

    tmp_uplift = conversion_for_plan - conversion_for_ref
    return tmp_uplift


@jit(nopython=True)
def _expected_conversion_rates(data_class, data_score, data_group,
                               smoothing=0):
    """This function estimates the expected conversion rates for all
    possible splits in the data and returns a list of these.

    E.g. the average of this list is the expected value for the
    conversion rate if you have _no_ prior preference on the split. It can
    also be used for visualization.

    Args:
    (See other functions in this package for details.)
    data_class (numpy.array([bool]))
    data_score (numpy.array([float]))
    data_group (numpy.array([bool]))
    smoothing (float): Used for estimation of conversion rates.

    Note:
    Old version could not possibly be used for estimation of E_r in
    a dataset with millions of samples. Estimating expected_uplift() for just
    one such takes seconds..
    """
    # Order all (this is used in the loop)
    n_samples = len(data_group)
    data_idx = np.argsort(data_score)[::-1]  # Sort in descending order.
    data_score = data_score[data_idx]
    data_group = data_group[data_idx]
    data_class = data_class[data_idx]

    # Initial counts: no treated samples
    conversions = []  # List of conversion rates for treatment plan and k
    treatment_goals = 0
    treatment_samples = 0
    control_goals = np.sum(data_class[~data_group])
    control_samples = np.sum(~data_group)
    # NUMERIC STABILITY?
    conversions.append((control_goals + smoothing) / max(1, control_samples + 2 * smoothing))
    # Needs to be averaged at end.
    previous_score = np.finfo(np.float32).min
    k = 0  # Counter for how many samples are "treated" of N (n_samples)
    first_iteration = True
    tmp_treatment_samples = 0
    tmp_treatment_goals = 0
    tmp_control_samples = 0
    tmp_control_goals = 0
    tmp_n_samples = 0

    for item_class, item_score, item_group in zip(data_class, data_score, data_group):
        if item_score == previous_score:
            # Add items to counters
            tmp_treatment_samples += int(item_group)  # If item is 'treatment group'
            tmp_treatment_goals += int(item_group) * item_class
            tmp_control_samples += int(~item_group)  # If item is 'control group'
            tmp_control_goals += int(~item_group) * item_class
            tmp_n_samples += 1  # One more sample to handle.
        else:
            if not first_iteration:
                # Not first iteration. Handle all equally scoring samples.
                for i in range(1, tmp_n_samples + 1):  # 0-indexing...
                    # We want to treat every equally scoring samples as the average of
                    # all equally scoring samples (i.e. expectation).
                    # Remember that control samples should be
                    # subtracted from the total and treatment
                    # samples added.
                    # Fraction of samples our model would like to treat:
                    p_t = (k - tmp_n_samples + i) / n_samples
                    # Fraction of samples our model would _not_ like to treat:
                    p_c = (n_samples - k + tmp_n_samples - i) / n_samples

                    tmp_t_goals = treatment_goals + i * tmp_treatment_goals / tmp_n_samples
                    tmp_t_samples = treatment_samples + i * tmp_treatment_samples / tmp_n_samples
                    # max() is here only to deal with the case where there are zero samples. This
                    # corresponds to estimating the conversion rate of zero samples to zero.
                    tmp_t_conversion = (tmp_t_goals + smoothing) /\
                                       max(1, tmp_t_samples + 2 * smoothing)

                    tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_samples
                    tmp_c_samples = control_samples - i * tmp_control_samples / tmp_n_samples
                    tmp_c_conversion = (tmp_c_goals + smoothing) /\
                                       max(1, tmp_c_samples + 2 * smoothing)
                    # The expected conversion rate when the i first samples should be treated
                    # is a weighted average of the two above:
                    conversion_rate = p_t * tmp_t_conversion + p_c * tmp_c_conversion
                    conversions.append(conversion_rate)
                # Add all samples as integers to treatment and control counters (not tmp)
                treatment_goals += tmp_treatment_goals
                treatment_samples += tmp_treatment_samples
                control_goals -= tmp_control_goals
                control_samples -= tmp_control_samples

            # Reset counters and add new item
            tmp_treatment_samples = int(item_group)  # If item is 'treatment group'
            tmp_treatment_goals = int(item_group) * item_class
            tmp_control_samples = int(~item_group)  # If item is 'control group'
            tmp_control_goals = int(~item_group) * item_class
            tmp_n_samples = 1
            previous_score = item_score
            first_iteration = False
        k += 1
    # Handle last samples
    for i in range(1, tmp_n_samples + 1):  # 0-indexing...
        # Remember that control samples should be
        # subtracted from the total and treatment
        # samples added. goal == conversion... conversion -> conversion_rate
        # Fraction of samples our model would like to treat:
        p_t = (k - tmp_n_samples + i) / n_samples
        # Fraction of samples our model would _not_ like to treat:
        p_c = (n_samples - k + tmp_n_samples - i) / n_samples

        tmp_t_goals = treatment_goals + i * tmp_treatment_goals / tmp_n_samples
        tmp_t_samples = treatment_samples + i * tmp_treatment_samples / tmp_n_samples
        # max() is here only to deal with the case where there are zero samples. This
        # corresponds to estimating the conversion rate of zero samples to zero.
        tmp_t_conversion = tmp_t_goals / max(1, tmp_t_samples)

        tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_samples
        tmp_c_samples = control_samples - i * tmp_control_samples / tmp_n_samples
        tmp_t_conversion = (tmp_t_goals + smoothing) / max(1, tmp_t_samples + 2 * smoothing)

        tmp_c_goals = control_goals - i * tmp_control_goals / tmp_n_samples
        tmp_c_samples = control_samples - i * tmp_control_samples / tmp_n_samples
        tmp_c_conversion = (tmp_c_goals + smoothing) / max(1, tmp_c_samples + 2 * smoothing)
        # The expected conversion rate when the i first samples should be treated
        # is a weighted average of the two above:
        conversion_rate = p_t * tmp_t_conversion + p_c * tmp_c_conversion
        conversions.append(conversion_rate)
    return conversions


def auuc_metric(data_class, data_score, data_group,
                ref_plan_type='rand',
                smoothing=0,
                testing=False):
    """This is a function for estimating the expected uplift for some treatment
    plan with respect to some other reference treatment plan _given_ no
    prior preference on fraction of samples to be treated. In a sense, this
    is the change in conversion rate you should expect given your treatment
    plan if you cannot say what fraction of samples should be treated.
    This is more or less equivalent to Jarosewicz's (2012) definition of AUUC.

    Args:
    data_class (np.array([bool]))
    data_score (np.array([float]))
    data_group (np.array([bool]))
    ref_plan_type (str) in {'no_treatments', 'all_treatments', 'rand', 'zero'}
    testing (bool): If True, the function uses expected_conversion_rate to estimate
    every point on the curve. This is extremely slow and not recommended.

    Notes:
    With "smoothing=0", this function estimates the conversion rate for
    treatment or conversion samples to zero if there are no samples to estimate
    from. This should be only a minor problem, though, that introduces close to
    1/N * conversion_rate error.
    """

    if testing:
        warnings.warn("auuc_metric(testing=True) is for testing purposes only!"+\
                      "The code is _very_ slow!")
        # Here for testing purposes!
        e_uplift = 0
        conversions = []
        n_samples = len(data_group)
        for i in range(n_samples + 1):
            conversion = expected_conversion_rate(data_class, data_score, data_group, i, smoothing)
            conversions.append(conversion)
            e_uplift += 1 / (n_samples + 1) *\
                        expected_uplift(data_class, data_score, data_group, i, ref_plan_type=ref_plan_type)
    else:
        conversions = _expected_conversion_rates(data_class, data_score, data_group,
                                                 smoothing=smoothing)
        if ref_plan_type == 'no_treatments':
            ref_conversion = conversions[0]
            # Would be more efficient to do the subtraction only once.
            uplifts = [conversion - ref_conversion for conversion in conversions]
        elif ref_plan_type == 'all_treatments':
            ref_conversion = conversions[-1]
            uplifts = [conversion - ref_conversion for conversion in conversions]
        elif ref_plan_type == 'rand':
            conversion_0 = conversions[0]
            conversion_1 = conversions[-1]
            n_samples = len(data_class)
            # Simply subtracting the average of the two above
            # from the mean should be enough.
            uplifts = [conversion - i / n_samples * conversion_1 - (n_samples - i) /\
                       n_samples * conversion_0 \
                       for i, conversion in zip(range(len(data_class) + 1), conversions)]
        elif ref_plan_type == 'zero':
            # This corresponds to estimating E_r over conversion rates
            uplifts = conversions
        else:
            raise Exception("Illegal ref_plan_type")
        # E_r is simply the mean of these uplifts:
        e_uplift = np.mean(uplifts)
    return e_uplift


def plot_conversion_rates(data_class, data_score, data_group, file_name='conversion.png'):
    """Function for plotting conversion rates vs. treatment rates for treatment
    rate in [0, 1]. This is almost equivalent to the uplift-curve defined by
    Jaskowski & Jarosewicz (2012) with the difference that they subtract the
    baseline conversion (conversion with no treatments) from all conversion
    rates. The plot has the same shape as the uplift-curve, only the y-axis
    differs.

    Args:
    ...
    file_name (str): Name of the file where you want the plot stored. Include
     .png-suffix.
    """
    conversions = _expected_conversion_rates(data_class, data_score, data_group)
    # Plot conversion rates:
    plt.plot([100 * x / len(conversions) for x in range(0, len(conversions))],\
             [100 * item for item in conversions])
    # Add random line:
    plt.plot([0, 100], [100 * conversions[0], 100 * conversions[-1]])
    plt.xlabel('Fraction of treated samples [%]')
    plt.ylabel('Conversion rate [%]')
    plt.title('Conversion rate vs. treatment rate')
    plt.savefig(file_name)
    plt.close()
    return()


def plot_uplift_curve(data_class, data_score, data_group, file_name='uplift_curve.png', revenue=None):
    """Function for plotting uplift vs. treatment rate following Jaskowski &
    Jarosewicz (2012). Very similar to plot_conversion_rates().
    This might be preferable if you want to highlight the change rather than the
    complete picture. E.g. if your model increases E_r(conversion rate) from
    3.1% to 3.15%, this is hardly relevant for the use case at hand. However
    if you are doing algorithm development, this difference might be interesting
    and then you might be better off studying uplift rather than conversion rates.

    Args:
    ...
    file_name (str): Name of the file where you want your uplift curve stored.
    revenue (float): Estimated revenue of one conversion. If 'revenue' is not
     None and is a floating point number, it will be used to estimate the
     incremental revenues of the uplift model. Otherwise the uplift curve will
     plot the increase in conversion rate as function of treatment rate. This
     is the default functionality.

    """
    conversions = _expected_conversion_rates(data_class, data_score, data_group)
    n_splits = len(conversions)
    # Conversion rate with no treatments
    conversion_0 = conversions[0]
    # Conversion rate with only treatments
    conversion_1 = conversions[-1]
    if revenue is not None:
        tmp = revenue
    else:
        tmp = 1
    plt.plot([100 * x / n_splits for x in range(0, n_splits)],
             [tmp * 100 * (conversion - conversion_0) for conversion, x in
              zip(conversions, range(len(conversions)))])
    # Add line for "random" model:
    plt.plot([0, 100], [0, tmp * 100 * (conversion_1 - conversion_0)])
    if revenue is not None:
        plt.ylabel('Cumulative revenue increase')
    else:
        plt.ylabel('Uplift [%]')
    plt.xlabel('Fraction of treated samples [%]')
    plt.title('Uplift curve')
    plt.savefig(file_name)
    plt.close()
    return()


def plot_base_probability_vs_uplift(data_probability,
                                    data_uplift,
                                    file_name='base_prob_vs_uplift.png',
                                    k=100000):
    """This function plots the predicted conversion probability vs. the
    predicted uplift. Note that the probabilities are not exactly "centered."
    This function was mostly for testing purposes.
    This could in principle show if there is some group that can be identified
    with response modeling (i.e. only predicting conversion probability, not
    uplift!) that is better or worse affected than the average user. This
    would enable you to do uplift modeling for that group but using a simpler
    approach with only response modeling.

    Args:
    data_probability (np.array([float, ...])): Vector of predicted conversion
     probabilities.
    file_name (str): Name of file where the plot should be stored.
    k (int): Size of the sliding window to smooth the uplift predictions.
    """
    # 'k' sets the size of the sliding window.
    idx = np.argsort(data_probability)  # Sort ascending
    data_probability = data_probability[idx]
    # Create sliding average
    probability_tmp = np.array([np.mean(data_probability[i:(i + k)]) for i in range(len(data_probability) - k)])
    data_uplift = data_uplift[idx]
    uplift_tmp = np.array([np.mean(data_uplift[i:(i + k)]) for i in range(len(data_uplift) - k)])
    plt.plot(probability_tmp, uplift_tmp)
    plt.xlabel('p(y|do(t=0))')
    plt.ylabel('p(y|x, do(t=1)) - p(y|x, do(t=0))')
    plt.title('Predicted conversion probability vs. uplift')
    plt.savefig(file_name)
    plt.close()
    return


def plot_uplift_vs_base_probability(data_probability,
                                    data_uplift,
                                    file_name='uplift_vs_base_prob.png',
                                    k=100000):
    """Function for plotting uplift vs. conversion probability. If this
    plot even vaguely resemples a 'V', then it indicates that there are
    samples where the uplift could not be simplified and derived from a
    response model. In technical terms, it implies that the uplift is not
    an injective function of the conversion probability. This is mostly
    a sanity check.

    Args:
    data_probability (np.array([float, ...])): Vector with predicted
    conversion probabilities for all samples in vector.
    data_uplift (np.array([float, ...])): Vector with predicted uplift
    probabilities for all samples.
    file_name (str): Name of file where plot is to be stored.
    k (int): Size of sliding window for smoothing of uplift

    Notes: The sliding window is for uplift predictions. There is no reason
    to assume that they would behave particularly nicely w.r.t. the conversion
    probability, hence the sliding window is necessary to make the graph
    smooth.
    """
    # 'k' sets the size of the sliding window.
    idx = np.argsort(data_uplift)  # Sort ascending
    data_probability = data_probability[idx]
    # Create sliding average
    probability_tmp = np.array([np.mean(data_probability[i:(i + k)]) for i in range(len(data_probability) - k)])
    data_uplift = data_uplift[idx]
    uplift_tmp = np.array([np.mean(data_uplift[i:(i + k)]) for i in range(len(data_uplift) - k)])
    plt.plot(uplift_tmp, probability_tmp)
    plt.ylabel('p(y|do(t=0))')
    plt.xlabel('p(y|x, do(t=1)) - p(y|x, do(t=0))')
    plt.title('Predicted uplift vs. conversion probability')
    plt.savefig(file_name)
    plt.close()
    return


@jit(nopython=True)
def _qini_points(data_class,
                 data_score,
                 data_group):
    """Auxiliary function for qini_coefficient().
    If we want some form of visualization of qini-coefficient
    (i.e. the standard graph), we should build on this.

    Args:
    data_class (numpy.array([bool]))
    data_score (numpy.array([float]))
    data_group (numpy.array([bool])): True indicates that sample
     belongs to the treatment-group.
    """
    # Order data in descending order:
    data_idx = np.argsort(data_score)[::-1]
    data_class = data_class[data_idx]
    data_score = data_score[data_idx]
    data_group = data_group[data_idx]

    # Set initial values for counters etc:
    qini_points = []
    # Normalization factor (N_t / N_c):
    n_factor = np.sum(data_group) / np.sum(~data_group)
    control_goals = 0
    treatment_goals = 0
    score_previous = np.finfo(np.float32).min
    tmp_n_samples = 1  # Set to one to allow division in first iteration
    tmp_treatment_goals = 0
    tmp_control_goals = 0
    for item_class, item_score, item_group in\
        zip(data_class, data_score, data_group):
        if score_previous != item_score:
            # If we have a 'new score', handle the samples
            # currently stored as counts...
            for i in range(1, tmp_n_samples + 1):
                # Turns out adding the zeroeth item is pointless.
                # Oh, well... it does not affect a thing.
                tmp_qini_point = (treatment_goals + i * tmp_treatment_goals /
                                  tmp_n_samples) -\
                                  (control_goals + i * tmp_control_goals /
                                   tmp_n_samples) * n_factor
                qini_points.append(tmp_qini_point)
            # Add tmp items to vectors before resetting them
            treatment_goals += tmp_treatment_goals
            control_goals += tmp_control_goals
            # Reset counters
            tmp_n_samples = 0
            tmp_treatment_goals = 0
            tmp_control_goals = 0
            score_previous = item_score
        # Add item to counters:
        tmp_n_samples += 1
        tmp_treatment_goals += int(item_group) * item_class
        tmp_control_goals += int(~item_group) * item_class

    # Handle remaining samples:
    for i in range(1, tmp_n_samples + 1):
        tmp_qini_point = (treatment_goals + i * tmp_treatment_goals /
                          tmp_n_samples) -\
                          (control_goals + i * tmp_control_goals /
                           tmp_n_samples) * n_factor
        qini_points.append(tmp_qini_point)

    # Make list into np.array:
    qini_points = np.array(qini_points)
    return qini_points


@jit(nopython=True)
def qini_coefficient(data_class, data_score, data_group):
    """Function for calculating the qini-coefficient of some data.
    Note that this does not implement the function as described
    by Diemert & al. as their normalization introduced a flaw.
    This version follows Radcliffe (who had included it
    in a different (but correct!) format. He, after all,
    introduced the definition of it in the first place!

    Args:
    data_class (numpy.array([bool]))
    data_score (numpy.array([float]))
    data_group (numpy.array([bool]))
    output_type (str) in {'coefficient', 'points'}. The second option returns
     a list of points on the qini-curve for e.g. plotting.

    # Handle equally scoring samples?
    # ...this also makes obvious what to do with equally
    # scoring samples (Expected value!), although that is not
    # that interesting as we will ditch this meteric anyway.
    """
    # Numba does not support warnings. Hence using print instead:
    print("The qini-coefficient is an outdated metric. Use other " +\
          "metrics in this package instead")

    qini_points = _qini_points(data_class, data_score, data_group)
    numerator = np.sum(qini_points)

    # Create artificial "optimal" ordering (that maximizes this
    # function) to estimate the denominator.
    new_data_group = np.array(([True] * np.sum(data_group)) +\
                              ([False] * np.sum(~data_group)))

    new_data_class = np.array(([True] * np.sum(data_class[data_group])) +
                              ([False] * np.sum(~data_class[data_group])) +
                              ([False] * int(np.sum(~data_class[~data_group]))) +
                              ([True] * int(np.sum(data_class[~data_group]))))

    # Score array so that first sample have highest score. This is just to get
    # the sorting not to change the order for estimation of "optimal ordering."
    new_data_score = np.array([i for i in range(len(data_group))][::-1])
    new_qini_points = _qini_points(data_class=new_data_class,
                                   data_score=new_data_score,
                                   data_group=new_data_group)
    denominator = np.sum(new_qini_points)
    # Calculate the qini-coefficient:
    result = numerator / denominator
    return result


# Naming?!? Statistical test.
def test_for_expected_conversion_rate(N_t1, N_c1,
                                      k_t1, n_t1, k_c1, n_c1,
                                      N_t2, N_c2,
                                      k_t2, n_t2, k_c2, n_c2,
                                      size=100000,
                                      check_time=False):
    """Statistical test for expected conversion rate. This is a bayesian
    test, although the maths should perferably be gone through once more.
    Plate-diagrams should help. This is for the expected conversion rate,
    not the expected conversion rate given no preference on split.
    Change naming?
    Basically this is a test for whether two treatment models produce
    conversion rates that are different to a statistically significant
    degree. This is similar to a bayesian test for difference in conversion
    rates where E(p_1 > p_2), i.e. an integral, is estimated using Monte
    Carlo simulation.

    Args:
    N_t1 (int): Number of samples that model_1 would like to treat.
    N_c1 (int) Number of samples that model_1 would _not_ like to treat.
    k_t_1 (int): Number of samples that model_1 would like to target and
     that were in fact targeted.
    n_t_1 (int): Number of samples where the desired treatment from model_1
     matches what was is present in the data, i.e. where the prediction
     matches what was actually done.
    *2 (*): Similar as above, but for model_2.
    check_time (bool): Indicator whether the function should be timed. This
     is for testing purposes.

    Notes:
    This function cannot be Numba-optimized as numba does not support Scipy.
    """
    if check_time:
        import time
        t_1 = time.clock()
    # First model:
    samples_1_1 = beta.rvs(N_t1 + 1, N_c1 + 1, size=size)
    samples_1_2 = beta.rvs(k_t1 + 1, n_t1 - k_t1 + 1, size=size)
    samples_1_3 = beta.rvs(k_c1 + 1, n_c1 - k_c1 + 1, size=size)
    # Item vice product
    tmp_1 = samples_1_1 * samples_1_2 + (1 - samples_1_1) * samples_1_3
    # Second model:
    samples_2_1 = beta.rvs(N_t2 + 1, N_c2 + 1, size=size)
    samples_2_2 = beta.rvs(k_t2 + 1, n_t2 - k_t2 + 1, size=size)
    samples_2_3 = beta.rvs(k_c2 + 1, n_c2 - k_c2 + 1, size=size)
    tmp_2 = samples_2_1 * samples_2_2 + (1 - samples_2_1) * samples_2_3
    prob = sum(tmp_1 > tmp_2) / size
    if check_time:
        print("Time: {}".format(time.clock() - t_1))
    return prob


@jit(nopython=True)
def _euce_points(data_class, data_prob, data_group,
                 k=100):
    """Auxiliary function for expected_uplift_calibration_error().
    This one is numba-optimized. This could also be used for visualization.

    data_class (numpy.array([bool]))
    data_prob (numpy.array([float])): Predicted change in conversion
     probability for each sample.
    data_group (numpy.array([bool]))
    k (int): Number of groups to split the data into for estimation.
    """
    # Doesn't matter if the sorting is ascending or descending.
    idx = np.argsort(data_prob)
    n_samples = len(data_prob)
    expected_errors = []
    # data_class = np.array([bool(item) for item in data_class])
    for i in range(k):
        tmp_idx = idx[int(n_samples / k * i):int((1 + i) * n_samples / k)]
        treatment_goals = np.sum(data_class[tmp_idx][data_group[tmp_idx]])
        treatment_samples = np.sum(data_group[tmp_idx])
        control_goals = np.sum(data_class[tmp_idx][~data_group[tmp_idx]])
        control_samples = np.sum(~data_group[tmp_idx])
        # Sanity check:
        assert treatment_samples + control_samples == len(tmp_idx), \
            "Error in estimation of expected calibration rate"
        assert treatment_goals + control_goals == np.sum(data_class[tmp_idx]),\
            "Error in estimation of expected calibration rate"
        uplift_in_data = (treatment_goals / treatment_samples) - (control_goals / control_samples)
        estimated_uplift = np.mean(data_prob[tmp_idx])
        expected_errors.append(np.abs(uplift_in_data - estimated_uplift))

    # Make numba-compatible:
    return np.array(expected_errors)


def expected_uplift_calibration_error(data_class, data_prob, data_group,
                                      k=100, verbose=False):
    """Function for estimating the expected calibration error and maximum
    calibration error for uplift. This is an extension of the ECE and MCE
    presented by Naeini & al. in 2015 (their metrics focused on response
    calibration, ours on uplift calibration).

    data_class (numpy.array([bool]))
    data_prob (numpy.array([float])): Predicted change in conversion
     probability for each sample.
    data_group (numpy.array([bool]))
    k (int): Number of groups to split the data into for estimation.
    """

    # Sanity check
    if k > len(data_class):
        raise Exception("k needs to be smaller than N!")

    try:
        expected_errors = _euce_points(data_class, data_prob,
                                       data_group, k=k)
    except Exception as e:
        print("******************************************")
        print("ERROR: Failed to run uplift_metrics._euce_points: %s" % e)
        print("******************************************")
        expected_errors = [float("nan"), float("nan")]
    euce = np.mean(expected_errors)
    muce = np.max(expected_errors)
    if verbose:
        print("Expected uplift calibration error: {}".format(euce))
        print("Maximum uplift calibration error: {}".format(muce))
    return (euce, muce)
