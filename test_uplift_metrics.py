"""
Tests for metrics for uplift-modeling.

Currently contains:
-tests for estimating conversion rate given some treatment plan
-tests for uplift
-tests for plotting

Perhaps add test for
-expected uplift with random reference treatment plan

"""
import warnings
import pytest
import numpy as np
import uplift_metrics


def test_1(i, smoothing=.5):
    """Test for 'expected_conversion_rate'

    Tests:
    -tie handling,
    -zero and all -cases
    -normal operations for i in {2, 3}
    """
    data_class = np.array([True, False, False, True, True, False, True])
    data_score = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.6, 0.7])
    data_group = np.array([True, False, True, False, False, True, True])
    tmp = uplift_metrics.expected_conversion_rate(data_class, data_score, data_group, i, smoothing)
    return tmp


def test_2(ref_plan_type):
    """Test value of expected uplift

    Also, test that efficient version produces identical results with
    brute force approach (testing=True).
    """
    data_class = np.array([True, False, False, True, True, False, True])
    data_score = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.6, 0.7])
    data_group = np.array([True, False, True, False, False, True, True])

    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group, ref_plan_type)
    tmp2 = uplift_metrics.auuc_metric(data_class, data_score, data_group, ref_plan_type,
                                      testing=True)
    assert tmp == pytest.approx(tmp2), "Error in test 2, expected uplift"
    return tmp


def test_4():
    """A test where all treatment samples are positive and all control
    samples zero.
    """
    data_class = np.array([True, False] * 3)
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([True, False] * 3)
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                     ref_plan_type='no_treatments')
    return tmp


def test_3():
    """Estimate qini-coefficient (not in use).

    Note that we estimate the areas as sums rather than interpolating
    between points.
    """
    data_class = np.array([True, False, False, True, True, False, True])
    data_score = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.6, 0.7])
    data_group = np.array([True, False, True, False, False, True, True])
    tmp = uplift_metrics.qini_coefficient(data_class, data_score, data_group)
    return tmp


def test_5():
    """A test where all treatment samples are positive and all control
    samples zero.
    """
    data_class = np.array([True, True] * 3)
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([True, False] * 3)
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                     ref_plan_type='no_treatments')
    return tmp


def test_6():
    """A test where all treatment samples are positive and all control
    samples zero.
    """
    data_class = np.array([False, False] * 3)
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([True, False] * 3)
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                         ref_plan_type='no_treatments')
    return tmp


def test_7():
    """A test where all treatment samples are positive and all control
    samples zero.
    Not in use.
    """
    data_class = np.array([False] * 4 + [True, False])
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([True, False] * 3)
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group, ref_plan_type='zero')
    uplift_metrics.plot_uplift(data_class, data_score, data_group, ref_plan_type='no_treatments')
    return tmp


def test_8():
    """A test where all treatment samples are positive and all control
    samples zero.
    """
    data_class = np.array([False] * 5 + [True])
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([False, True] * 3)
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                     ref_plan_type='no_treatments')
    uplift_metrics.plot_conversion_rates(data_class, data_score, data_group, file_name='tmp.png')
    return tmp


def test_9():
    """Test for optimal vs. suboptimal ordering of samples.

    The optimal ordering should score higher.
    """
    data_class = np.array([True, False, True, False, False, True])
    data_score = np.array([-i for i in range(6)])
    data_group = np.array([True, False, True, False, True, False])
    uplift_metrics.plot_conversion_rates(data_class, data_score, data_group, file_name='tmp.png')
    tmp_optimal = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                             ref_plan_type='no_treatments')
    # Change to sub-optimal ordering:
    data_class = np.array([True, True, True, False, False, False])
    tmp_sub_optimal = uplift_metrics.auuc_metric(data_class, data_score, data_group,
                                                     ref_plan_type='no_treatments')
    assert tmp_optimal > tmp_sub_optimal, "Error in test 9"
    return


def test_10():
    """A test for expected calibration error.
    """
    data_class = np.array([False, True] * 50 + [True] * 3)
    data_score = np.array([-i for i in range(103)])
    data_group = np.array([False, True, True, False] * 25 + [False] * 3)
    tmp = uplift_metrics.expected_uplift_calibration_error(data_class, data_score, data_group, k=10)
    return tmp


def test_11():
    """A test for testing the UpliftMetrics-class
    """
    data_class = np.array([False, True] * 50 + [True] * 3)
    data_score = np.array([-i / 100 for i in range(103)])
    data_group = np.array([False, True, True, False] * 25 + [False] * 3)
    uplift_metrics_tmp = uplift_metrics.UpliftMetrics(data_class, data_score, data_group,
                                                      test_name="1",
                                                      test_description="code test",
                                                      algorithm="None",
                                                      dataset='dummy data break csv format"?',
                                                      parameters='break 2 csv format ";"?',
                                                      euce_k=10)
    print(uplift_metrics_tmp)
    uplift_metrics_tmp.save_to_csv("code_test_results.csv")

    # Test that metrics in the object corresponds to results estimated using
    # separate functions:
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group, ref_plan_type='rand')
    assert tmp == pytest.approx(uplift_metrics_tmp.auuc), "Error in class AUUC estimation"
    tmp = uplift_metrics.auuc_metric(data_class, data_score, data_group, ref_plan_type='zero')
    assert tmp == uplift_metrics_tmp.e_r_conversion_rate, \
        "Error in class E_r(conversion rate) estimation"
    tmp_0 = uplift_metrics.expected_conversion_rate(data_class, data_score,
                                                    data_group, k=0)
    assert tmp_0 == uplift_metrics_tmp.conversion_rate_no_treatments,\
        "Error in class estimation of conversion rate with no treatments"
    tmp_1 = uplift_metrics.expected_conversion_rate(data_class, data_score,
                                                    data_group, k=len(data_class))
    assert tmp_1 == uplift_metrics_tmp.conversion_rate_all_treatments,\
        "Error in class estimation of conversion rate with all treatments"
    assert np.mean([tmp_0, tmp_1]) == uplift_metrics_tmp.e_conversion_rate_random,\
        "Error in estimation of random conversion rate"
    tmp = uplift_metrics.expected_uplift_calibration_error(data_class, data_score,
                                                           data_group, k=10)
    assert tmp[0] == uplift_metrics_tmp.euce, "Error in class-estimation of EUCE"
    assert tmp[1] == uplift_metrics_tmp.muce, "Error in class-estimation of MUCE"
    assert len(data_class) == uplift_metrics_tmp.samples, "Error in estimation of #samples."
    assert len(np.unique(data_score)) == uplift_metrics_tmp.unique_scores, \
        "Error in counting #unique scores"
    
    

def run_tests():
    """Function for running selected tests in this file.

    """
    # Test 1, division by zero with smoothing=0:
    warnings.simplefilter("error", RuntimeWarning)
    try:
        # When this throws a RuntimeWarning, tmp does not get a value.
        tmp = test_1(6, 0)
    except RuntimeWarning:
        print("Error in test 1")
    warnings.resetwarnings()  # Undo the effects of the simplefilter above.
    # Test 2.1:
    # Values for different input (tie handling, zero case, basic estimation)
    res = [5 / 8, 9 / 14, 33 / 56, 1 / 2,
           33 / 70, 67 / 154, 11 / 28, 1 / 2]
    for i in range(8):
        tmp = test_1(i, .5)
        assert res[i] == pytest.approx(tmp), "Error in test_1.1, i = {}".format(i)
    # Test 2.2:
    # Value of expected uplift given no prior info on preference for split.
    assert test_2(ref_plan_type='no_treatments') == pytest.approx(-0.1531675170068027), \
        "Error in test 2.2.1"
    assert test_2(ref_plan_type='all_treatments') == pytest.approx(0.013499149659863985), \
        "Error in test 2.2.2"
    assert test_3() == pytest.approx(-24 / 81), "Error in test 3, qini-coefficient"
    assert test_4() == pytest.approx(.5), "Error in test 4"
    assert test_5() == pytest.approx(0), "Error in test 5"
    assert test_6() == pytest.approx(0), "Error in test 6"
    # test_7()  # Test for the 'zero' reference plan, not implemented.
    assert test_8() == pytest.approx(1 / 3 / 7), "Error in test 8"
    # Sanity check: value of optimal vs. suboptimal ordering:
    test_9()
    # Sanity check: check that expected_uplift_calibration_error runs.
    test_10()
    # Just run the uplift-metrics class:
    test_11()
    print("All tests passed.")
    return


if __name__ == '__main__':
    run_tests()
