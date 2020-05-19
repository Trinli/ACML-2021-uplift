
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class DCLogisticRegression():
    """
    Version of logistic regression that handles uplift-related stuff.
    """
    def __init__(self):
        self.t_model = LogisticRegression(solver='lbfgs')
        self.c_model = LogisticRegression(solver='lbfgs')

    def fit(self, X_c, y_c, X_t, y_t):
        """
        Method for fitting a double-classifier.

        Args:
        X_c (numpy.array): Features for control data
        y_c (numpy.array): Label for control data
        ...
        """
        self.t_model.fit(X=X_t, y=y_t)
        self.c_model.fit(X=X_c, y=y_c)

    def predict_uplift(self, X):
        """
        Method for predicting uplift.

        Args:
        X (np.array): Features for samples.
        """
        t_prediction = self.t_model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_t_idx = np.where(self.t_model.classes_ == 1.0)[0][0]
        t_pred = t_prediction[:, true_t_idx].astype(np.float64)
        c_prediction = self.c_model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_c_idx = np.where(self.c_model.classes_ == 1.0)[0][0]
        c_pred = c_prediction[:, true_c_idx].astype(np.float64)
        return t_pred - c_pred


class ClassVariableTransformation():
    """
    Version of class-variable transformation that is just a plain model.
    """
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs')

    def fit(self, X, z):
        """
        Fitting the model, assumes class-variable transformation as input.

        Args:
        X (): Features
        z (np.array): Class-variable transformed dependent variable.
        """
        self.model.fit(X=X, y=z)

    def predict_uplift(self, X):
        """
        Function for predicting uplift, i.e. change in y.

        Args:
        X (np.array): Features
        """
        tmp_prediction = self.model.predict_proba(X)
        # Figure out which column is probability predictions for True
        true_idx = np.where(self.model.classes_ == 1.0)[0][0]
        return 2.0 * tmp_prediction[:, true_idx].astype(np.float64) - 1.0


class CVTRandomForest(ClassVariableTransformation):
    """
    Class-variable transformation with random forest.
    """
    def __init__(self):
        self.model = RandomForestClassifier()


class DCRandomForest(DCLogisticRegression):
    """
    Double-classifier approach with random forest.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.t_model = RandomForestClassifier()
        self.c_model = RandomForestClassifier()
