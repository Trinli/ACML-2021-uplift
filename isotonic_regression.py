"""
This is an implementation of calibration using isotonic regression
for uplift modeling.

UpliftIsotonicRegression uses Athey & Imbens' revert-label aproach (2016?).
This approach does not guarantee equally many treatment and
control samples in one bin, but the PAVA will lead to a result where the
bins approximately contain equally many treatment and control samples by
simply merging bins until the predicted probabilities map to scores
monotonically.
-This is theoretically justified.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class UpliftIsotonicRegression(object):
    """
    Variant of isotonic regression-based calibration for
    uplift modeling using Athey & Imbens' revert-label.
    """
    def __init__(self, y_min=-.999, y_max=.999):
        """
        Args:
        y_min (float): Smallest uplift this model can predict
        y_max (float): Largest uplift this model can predict
        """
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max,
                                        out_of_bounds='clip')

    def fit(self, x_vec, y_vec, t_vec):
        """
        Args:
        x_vec (np.array): 1d array of scores
        y_vec (np.array): Label, True indicates positive
        t_vec (np.array): group, True indicates treatment group
        """
        # 1. Define 'r' as the revert label
        p_t = sum(t_vec) / len(t_vec)
        # This does not seem right:
        r_vec = np.array([y_i * (t_i - p_t) / (p_t * (1 - p_t)) for\
                          y_i, t_i in zip(y_vec, t_vec)])

        # 2. Run isotonic regression on the new problem
        self.model.fit(x_vec, r_vec)

    def predict_uplift(self, x_vec):
        """
        Method for predicting calibrated change in probability.

        Args:
        x_vec (np.array): Scores to convert to probabilities
        """
        return self.model.predict(x_vec)
