"""
Classes for handling datasets for uplift experiments.

Notes:
Support for ordinal variables could perhaps be added. Somewhat involved
 as labels needs to be mapped to values (e.g. likert-scale).
Perhaps add tests for load_data.
Support for other dependent variables than simply 0/1 could be added
 following the same logic used with treatment labels (t_labels).
Perhaps expand the documentation for the data format.
Not tested for h5py-files.
"""

import h5py
import csv
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

# Data format as indices for features, label, and group and
# some other info. This is essentially a configuration.
# This format is used for Criteo-uplift data:
DATA_FORMAT = {'X_idx': [i for i in range(12)],  # indices for feature columns
               'continuous_idx': [i for i in range(12)],  # indices for continuous features
               'categorical_idx': [],  # indices for categorical indices
               'y_idx': 13,  # index for the dependent variable column
               't_idx': 12,  # index for the treatment label column
               't_labels': ['1', '0'],  # Treatment and control labels.  
               'random_seed': 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.

# This format is for the Hillstrom Mine that data -challenge dataset
HILLSTROM_FORMAT_1 = {'file_name':
                      'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv',
                      # Note that many of the features are categorical
                      'X_idx': [i for i in range(8)],
                      # There are a few binary features and a few categorical
                      # ones with multiple values.
                      'continuous_idx': [0, 2],
                      'categorical_idx': [1, 3, 4, 5, 6, 7],
                      # y = 9 corresponds to 'visit'. Conversion has too few positive.
                      'y_idx': 9,  # There is also a possibility to focus on spent money.
                      't_idx': 8,
                      # This format would include men's email as treatment group and
                      # no e-mail as control group.
                      't_labels': ['Mens E-Mail', 'No E-Mail'],  # 'Womens E-Mail'],
                      'random_seed': 6,
                      'headers': True,
                      'normalization': 'v3',
                      'data_type': 'float32'}  # Datatype does not work.


class DatasetCollection(object):
    """
    Class for handling dataset related operations (loading, normalization,
     dummy-coding, access, group-subsetting, undersampling).

    Methods:
    __init__(file_name (str), mode={None, 'test'}):
     Initialization loads data from specified csv or h5py -file.
    _normalize_data(): Normalizes data. Default normalization is 'v3'
     specified in DATA_NORMALIZATION and includes centering and setting
     of variance to 1 for features.
    _load_data(): Imports both csv and h5py-files
    _extract_data(): Reads in features, class-labels, and treatment-label
     following the data specified in self.data_format.
    _create_subsets(): Creates training, validation, and testing sets with
     splits 1/2, 1/4, 1/4, and further adds a second training, and two
     validation sets with 6/16, 3/16, 3/16 splits. This leaves the testing
     set untouched.
    __add_set(): Auxiliary function for _create_subsets().
    __getitem__(): Method for accessing all data once loaded in the initialization.
    _undersample(): Method for undersampling data both to 1:1 and 1:1:1:1.
    __subset_by_group(): Method for selecting only treatment, control, or all data.
    """
    def __init__(self, file_name,
                 data_format=DATA_FORMAT, 
                 set_conversion_rate=None):
        """
        Method for initializing object of class DataSet that will handle
        all dataset related issues.

        Attributes:
        file_name (str): {'criteo-uplift.h5', 'criteo100k.csv', 'criteo-uplift.csv', ...}
        nro_features (int): number of features in dataset. Assumed to be in columns 0:nro_features.
        seed (int): random seed to use for split of data.

        Notes:
        Potentially dataloaders could also be stored in this object i a dict
         in the same way datasets are stored now. Would require carrying information
         on treatment/control.
        """
        self.file_name = file_name
        self.data_format = data_format
        self.nro_features = len(self.data_format['X_idx'])
        # Load data into array. All data is in csv or similar format.
        # Load data into self.X, self.y, self.t and self.z:
        # Do some light preprocessing (e.g. identify class label, features and group).
        # This function will parse the data into X, y, and t following the format
        # specified in self.data_Format.
        self._load_data(set_conversion_rate=set_conversion_rate)
        # Randomize
        self._shuffle_data()

        # Create empty dict for storing of usable datasets such as training set etc:
        self.datasets = {}
        # Populate self.datasets with predefined subsets:
        self._create_subsets()

    def _load_data(self, set_conversion_rate=None):
        """
        Method for loading data from file. Currently supports h5py-
        and csv-files.

        Maybe store the data into features (X), y, and t at this point already?
        Args:
        set_conversion_rate (float): If new_rate not None, then the new positive rate is defined
         using this. Positive samples are randomly dropped until the entire dataset
         exhibits a positive rate of new_rate (e.g. the range [.002, .005, .01, .02, .041]
         would be reasonable). The natural positive rate is something like .041 for the
         visit-label in the Criteo dataset.
        """
        tmp_data = []
        if self.file_name.endswith('.h5'):
            try:
                with h5py.File(self.file_name) as fh:
                    tmp_data = fh['data'][:]
            except:
                raise Exception("The file {} was not found in the working directory.".format(self.file_name))

        elif self.file_name.endswith('csv'):
            try:
                with open(self.file_name, "r") as handle:
                    file_ = csv.reader(handle)
                    for row in file_:
                        tmp_data.append(row)
            except:
                raise Exception("The file {} was not found in the working directory.".format(self.file_name))
            if self.data_format['headers']:
                # This becomes misleading as we reformat the data:
                # self.col_names = tmp_data[1]
                # Drop title row:
                tmp_data = tmp_data[1:]
            # Perhaps make class and treatment labels binary?
            # Everything is one array which prevents this.
        else:
            raise Exception("The file needs to be a .csv or .h5py -file.")

        # Parse data into suitable format following format specified in
        # self.data_format:
        tmp_X = None        
        for col, _ in enumerate(tmp_data[0]):
            # Parse into suitable format
            # Either it is continuous or categorical.
            # Some columns are also treatment-group labels etc ?!?
            # if column is feature:
            if col in self.data_format['X_idx']:
                if col in self.data_format['continuous_idx']:
                    tmp = np.array([[row[col]] for row in tmp_data])
                    tmp = tmp.astype(self.data_format['data_type'])
                    tmp = self._normalize_data(tmp)
                elif col in self.data_format['categorical_idx']:
                    tmp_col = [row[col] for row in tmp_data]
                    keys = np.unique(tmp_col)
                    tmp = np.zeros([len(tmp_col), len(keys)], dtype=
                                   self.data_format['data_type'])
                    for i, key in enumerate(keys):
                        for j, _ in enumerate(tmp_col):
                            if tmp_col[j] == key:
                                tmp[j, i] = 1
                # Add new features to tmp_array
                if tmp_X is None:
                    tmp_X = tmp
                else:
                    tmp_X = np.concatenate([tmp_X, tmp], axis = 1)
            elif col == self.data_format['t_idx']:
                # Binary group label
                # True indicates treatment group
                tmp = [row[col] for row in tmp_data]
                tmp_t = np.array([item == self.data_format['t_labels'][0] for
                                  item in tmp])
            elif col == self.data_format['y_idx']:
                tmp_y = np.array([row[col] for row in tmp_data])
                tmp_y = tmp_y.astype(np.bool_)
                # tmp_y = tmp_y.astype(self.data_format['data_type'])
        # Class-variable transformation:
        tmp_z = tmp_y == tmp_t

        # Keep only samples that belong to specified treatment groups:
        tmp = [row[self.data_format['t_idx']] for row in tmp_data]
        group_idx = [item in self.data_format['t_labels'] for item in
                     tmp]
        
        group_idx = np.array(group_idx)
        self.X = tmp_X[group_idx, :]
        self.y = tmp_y[group_idx]
        self.t = tmp_t[group_idx]
        self.z = tmp_z[group_idx]
        
        # Insert code here for changing positive rate.
        if set_conversion_rate is not None:
            # 1. Estimate current conversion rate
            tmp_rate = sum(self.y)/len(self.y)
            # 2. Estimate how many positive samples should be kept
            positive_y = int(set_conversion_rate / tmp_rate * sum(self.y))
            # 3. Randomly draw the amount of positive samples desired
            # Was 'y' boolean?
            tmp_pos_idx = np.random.choice([idx for idx, item in enumerate(self.y) if item],
                                           size=positive_y,
                                           replace=False)
            # 4. Identify all negative samples
            tmp_neg_idx = np.array([idx for idx, item in enumerate(self.y) if not item])
            # 5. Populate X, y, t, and z -vectors (matices).
            tmp_idx = np.concatenate((tmp_pos_idx, tmp_neg_idx))
            # Shuffle in place:
            np.random.shuffle(tmp_idx)
            self.X = self.X[tmp_idx, :]
            self.y = self.y[tmp_idx]
            self.t = self.t[tmp_idx]
            self.z = self.z[tmp_idx]
        
        # Print some statistics for the loaded data:
        print("Dataset {} loaded".format(self.file_name))
        print("\t\t\t#y\t#samples\tconversion rate")
        print("All samples", end='\t\t')
        print(sum(self.y), end='\t')
        print(len(self.y), end='\t')
        print(sum(self.y)/len(self.y))
        print("Treatment samples", end='\t')
        print(sum(self.y[self.t]), end='\t')
        print(sum(self.t), end='\t')
        print(sum(self.y[self.t]/sum(self.t)))
        print("Control samples", end='\t\t')
        print(sum(self.y[~self.t]), end='\t')
        print(sum(~self.t), end='\t')
        print(sum(self.y[~self.t]/sum(~self.t)))
        conversion_rate_treatment = sum(self.y[self.t])/sum(self.t)
        conversion_rate_control = sum(self.y[~self.t])/sum(~self.t)
        effect_size = (conversion_rate_treatment - conversion_rate_control) /\
                      conversion_rate_control
        print("Estimated effect size for treatment: {:.2f}%".format(effect_size * 100))

              
    def _shuffle_data(self):
        if self.data_format['random_seed'] is not None:
            print("Random seed set to {}.".format(self.data_format['random_seed']))
            # Set random seed to get same split for all experiments
            np.random.seed(self.data_format['random_seed'])
        shuffling_idx = np.random.choice([item for item in range(len(self.y))],
                                         len(self.y), replace=False)
        self.X = self.X[shuffling_idx, :]
        self.y = self.y[shuffling_idx]
        self.t = self.t[shuffling_idx]
        self.z = self.z[shuffling_idx]
        # self.r does not exist at this point yet
        # self.r = self.r[shuffling_idx]
        # Reset seed:
        np.random.seed(None)  # Uses time
        
    def _create_subsets(self):
        """
        Method for creating training, validation, and testing sets plus
        an additional training_set_2, validation_set_2a, and validation_set_2b
        to be used for deciding on early stopping when training neural networks.
        Training, validation, and testing sets are split 50:25:25, and
        training2, validation_set_2a, validation_set_2b, and testing set are
        split 6/16:3/16:3/16:4/16. Note that in both cases, the testing sets are
        identical in all aspects.
        """
        # Using a 50/25/25 split
        n_samples = self.X.shape[0]
        # Add usable datasets such as training set to self.datasets:
        self.__add_set('training_set', 0, n_samples // 2)
        self.__add_set('validation_set', n_samples // 2, n_samples * 3 // 4)
        self.__add_set('testing_set', n_samples * 3 // 4, n_samples)
        # Add also slightly different split that enables early stopping of
        # neural networks using separate validation set:
        self.__add_set('training_set_2', 0, n_samples * 6 // 16)
        self.__add_set('validation_set_2a', n_samples * 6 // 16, n_samples * 9 // 16)
        self.__add_set('validation_set_2b', n_samples * 9 // 16, n_samples * 12 // 16)
        tmp = self._undersample('training_set', '1:1')
        # Undersampled training set:
        self.datasets.update({'undersampled_training_set_11': tmp})
        tmp = self._undersample('training_set', '1:1:1:1')
        self.datasets.update({'undersampled_training_set_1111': tmp})
        tmp = self._undersample('training_set_2', '1:1')
        self.datasets.update({'undersampled_training_set_2_11': tmp})
        tmp = self._undersample('training_set_2', '1:1:1:1')
        self.datasets.update({'undersampled_training_set_2_1111': tmp})

    def _revert_label(self, y_vec, t_vec):
        """
        Class-variable transformation ("revert-label") approach
        following Athey & Imbens (2015).

        Args:
        y_vec (np.array([float])): Array of conversion values in samples.
        t_vec (np.array([bool])): Array of treatment labels for same samples.
        """
        N_t = sum(t_vec == True)
        N_c = sum(t_vec == False)
        # Sanity check:
        assert N_t + N_c == len(t_vec), "Error in sample count (_revet_label())."
        p_t = N_t / (N_t + N_c)
        def revert(y_i, t_i, p_t):
            return (y_i * (int(t_i) - p_t) / (p_t * (1 - p_t)))
        r_vec = np.array([revert(y_i, t_i, p_t) for y_i, t_i in zip(y_vec, t_vec)])
        return r_vec

    def __add_set(self, name, start_idx, stop_idx):
        """
        Auxiliary function for _create_subsets(). Adds usable datasets as dicts
        with X, y, t, and z. 'z' here refers to class-variable transformation
        following Jaskowski & Jaroszewicz (2012).
        """
        X_tmp = self.X[start_idx:stop_idx, :]
        y_tmp = self.y[start_idx:stop_idx]
        t_tmp = self.t[start_idx:stop_idx]
        z_tmp = y_tmp == t_tmp
        r_tmp = self._revert_label(y_tmp, t_tmp)
        self.datasets.update({name: {'X': X_tmp,
                                     'y': y_tmp,
                                     't': t_tmp,
                                     'z': z_tmp,
                                     'r': r_tmp}})
        
        
    def _normalize_data(self, vector):
        """
        Method for normalizing data.

        Attributes:
        version (str): {'v1', 'v2', 'v3'}
         -v1 and v2 are for testing purposes
         -v3 (default) centers data and sets variance to unit (1)
        """
        # This normalizes features as stated in Diemert & al.
        if self.data_format['normalization'] is None:
            pass
        elif self.data_format['normalization'] == 'v1':
            tmp = normalize(vector)
        elif self.data_format['normalization'] == 'v2':
            # Set variance to 1
            tmp = vector / np.std(vector)
        elif self.data_format['normalization'] == 'v3':
            # Set mean to 0 and variance to 1:
            tmp = vector - np.mean(vector)
            tmp = tmp / np.std(tmp)
        return tmp

    def _undersample(self, name, method):
        """
        Method that adds undersampled data as np.array into dataset dict
        together with 'X', 'y', and 't'.

        Args:
        name (str): Name of set to be added to
        # Where do we get the info on which set to undersample? Always
        # training_set?
        1:1 undersampling only for SVM double classifier (with separated
        (treatment and control sets)
        1:1:1:1 used together with CVT (any base learner)

        Notes:
        Only the '1:1' is theoretically sound in this method. Changing to 1:1:1:1
        is just a quick hack. k_undersampling() is better.
        """
        tmp = self.datasets[name]
        tot_samples = len(tmp['t'])
        if method == '1:1':
            # 1:1, treatment:control ratio
            # Smaller of these is the number of samples we want
            n_samples = np.int(np.min([sum(tmp['t']), sum(~tmp['t'])]))
            treatment_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                      if t])
            control_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                    if not t])
            # Shuffle:  -no random seed?
            np.random.shuffle(treatment_idx)
            np.random.shuffle(control_idx)
            idx = np.concatenate([treatment_idx[:n_samples],
                                  control_idx[:n_samples]])
        elif method == '1:1:1:1':
            print("Note that the 1:1:1:1 method is not theoretically sound. Perhaps deprecate?")
            # positive_treatment:negative_treatment:positive_control:
            # negative_control, 1:1:1:1
            # Looking for min of positive or negative classes in any group:
            n_samples = np.int(np.min([np.sum(tmp['t'] * tmp['y']),
                                       np.sum(~tmp['t'] * tmp['y']),
                                       np.sum(tmp['t'] * (tmp['y'] == 0)),
                                       np.sum(~tmp['t'] * (tmp['y'] == 0))]))
            pos_treatment_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                               tmp['t'], tmp['y'])
                                          if t & (y == 1)])
            neg_treatment_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                               tmp['t'], tmp['y'])
                                          if t & (y == 0)])
            pos_control_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                             tmp['t'], tmp['y'])
                                        if (not t) & (y == 1)])
            neg_control_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                             tmp['t'], tmp['y'])
                                        if (not t) & (y == 0)])
            # Shuffle so that [:n_samples] is a random sample:
            np.random.shuffle(pos_treatment_idx)
            np.random.shuffle(neg_treatment_idx)
            np.random.shuffle(pos_control_idx)
            np.random.shuffle(neg_control_idx)
            # Take #n_samples from each type and concatenate:
            idx = np.concatenate([pos_treatment_idx[:n_samples],
                                  neg_treatment_idx[:n_samples],
                                  pos_control_idx[:n_samples],
                                  neg_control_idx[:n_samples]])
        else:
            raise Exception("The defined undersampling method, " +
                            "{}, does not exist".format(method))
        # Shuffle index for good measure (prevent any idiosyncrasies in
        # algorithms to have weird effects)
        np.random.shuffle(idx)
        X = tmp['X'][idx, :]
        y = tmp['y'][idx]
        t = tmp['t'][idx]
        z = tmp['z'][idx]
        # Revert-label does not make sense together with undersampling.
        # (techically it is possible to calculate, but the normalization is
        # precisely there to make undersampling unnecessary.)
        # At minimum, it would need to be recalculated for a subset.
        return {'X': X, 'y': y, 't': t, 'z': z}


    def k_undersampling(self, k, group_sampling='11'):
        """
        Method returns a training set where the rate of positive samples
        is changed by a factor of k by either reducing the number of
        negative samples or increasing the number of positive samples.
        The method also changes the sampling rate of treatment vs. control
        samples to 1:1.
        This is suitable for class-variable transformation.

        Args:
        k (int): If None, a balanced k is deduced from the data. Otherwise
         this number will determine the change in positive rate in the data.
         group_sampling (str): 'natural' implies no change in group sampling
         rate, i.e. the number of samples in the treatment and control groups
         stay constant. 
         '11' indicates that there should be equally many treatment and
         control samples. This is useful with CVT and enforces 
         p(t=0) = p(t=1)).

        Notes:
        If k is very large the number of negative samples might drop to zero,
        or conversely if k is very small the number of positive samples might
        drop to zero. There is not a check for this implemented. The implementation
        ensures at least one negative sample is retained.
        """
        # Number of positives in treatment group:
        t_data = self['training_set', None, 'treatment']
        num_pos_t = sum(t_data['y'])
        # Find indices for all positive treatment samples
        pos_idx_t = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is True])
        num_neg_t = sum(~t_data['y'])
        # Find indices for all negative treatment samples:
        neg_idx_t = np.array([i for i, tmp  in enumerate(zip(self['training_set']['y'],
                                                             self['training_set']['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is True])
        num_tot_t = len(t_data['y'])
        
        c_data = self['training_set', None, 'control']
        num_pos_c = sum(c_data['y'])
        # Find indices for all positive control samples
        pos_idx_c = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is False])
        num_neg_c = sum(~c_data['y'])
        # Find indices for all negative control samples:
        neg_idx_c = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is False])
        num_tot_c = len(c_data['y'])
        # Adjust the total number of positive and negative samples in treatment and
        # control groups separately to change the positive rate by k:
        if k >= 1:
            num_neg_c_new = max(0, int(num_tot_c // k) - num_pos_c)
            num_neg_t_new = max(0, int(num_tot_t // k) - num_pos_t)
            num_pos_c_new = num_pos_c
            num_pos_t_new = num_pos_t
        elif k < 1 and k > 0:  # Aiming for k * pos/tot = pos_new / tot_new
            num_neg_c_new = num_neg_c  # stays constant
            num_pos_c_new = max(0, int(k * num_pos_c / num_tot_c * num_neg_c /\
                                       (1 - k * num_pos_c / num_tot_c)))
            num_neg_t_new = num_neg_t  # stays constant
            num_pos_t_new = max(0, int(k * num_pos_t / num_tot_t * num_neg_t /\
                                       (1 - k * num_pos_t / num_tot_t)))
        else:
            raise ValueError("k needs to be larger than 0")

        # Change number of samples to be picked in treatment or control group to
        # make num_tot_t == num_tot_c:
        if group_sampling == '11':
            num_tot_c_new = num_neg_c_new + num_pos_c_new
            num_tot_t_new = num_neg_t_new + num_pos_t_new
            if num_tot_c_new > num_tot_t_new:
                # Reduce number of control samples:
                coef = num_tot_t_new / num_tot_c_new
                num_neg_c_new = int(coef * num_neg_c_new)
                num_pos_c_new = int(coef * num_pos_c_new)
            else:
                # Reduce number of treatment samples:
                coef = num_tot_c_new / num_tot_t_new
                num_neg_t_new = int(coef * num_neg_t_new)
                num_pos_t_new = int(coef * num_pos_t_new)

        # Create indices for sampling:
        new_neg_c_idx = np.random.choice(neg_idx_c, size=num_neg_c_new, replace=False)
        new_neg_t_idx = np.random.choice(neg_idx_t, size=num_neg_t_new, replace=False)
        new_pos_c_idx = np.random.choice(pos_idx_c, size=num_pos_c_new, replace=False)
        new_pos_t_idx = np.random.choice(pos_idx_t, size=num_pos_t_new, replace=False)
        
        idx = np.concatenate([new_pos_t_idx, new_neg_t_idx,
                              new_pos_c_idx, new_neg_c_idx],
                             axis=0)
        # Shuffle in place for good measure
        np.random.shuffle(idx)
        tmp_X = self['training_set']['X'][idx, :]
        tmp_y = self['training_set']['y'][idx]
        tmp_t = self['training_set']['t'][idx]
        tmp_z = self['training_set']['z'][idx]
        # We will also need 'r' here now (MSE-gradient)!
        tmp_r = self._revert_label(tmp_y, tmp_t)
        
        return {'X': tmp_X, 'y': tmp_y, 'z': tmp_z, 't': tmp_t, 'r': tmp_r}
        

    def __getitem__(self, *args):
        """
        Shorthand method to access self.datasets' contents. This function will
        return either a numpy-array or a pytorch dataloader depending on parameters.

        Args:
        args[0] = name (str): name of key in data.datasets to access, e.g.
         'training_set'.
        args[1] = undersampling {None, '11', '1111'}: None causes no undersampling
         at all, '11' results in treatment and control groups being equally large,
         '1111' results in '11' and #positive and #negative in both groups to be
         equally large.
        args[2] = group {'all', 'treatment', 'control'}: 'all' and None both
         return all data. 'treatment' returns samples that were treated etc.

        Notes:
        *No solution for fetching data with CVT by Athey & Imbens (2015)
        """
        # Handle input arguments:
        group = 'all'  # Subset for treatment or control?
        if isinstance(args[0], str):  # Only name of dataset passed.
            name = args[0]
        elif isinstance(args[0], tuple):  # Multiple arguments were passed
            name = args[0][0]
            if len(args[0]) > 1:
                undersampling = args[0][1]
                if (undersampling is not None) and name == 'training_set':
                    name = 'undersampled_' + name + '_' + str(undersampling)
                elif undersampling is not None:
                    print("Currently no undersampled datasets for other " +
                          "than training set.")
                if len(args[0]) > 2:
                    group = args[0][2]
                    if len(args[0]) > 3:
                        raise Exception("Too many arguments.")
        else:
            raise Exception("Error in __getitem__()")

        # Store approproate data in tmp:
        if group == 'treatment':
            idx = self.datasets[name]['t']
            tmp = self.__subset_by_group(name, idx)
        elif group == 'control':
            # Negation of 't':
            idx = ~self.datasets[name]['t']
            tmp = self.__subset_by_group(name, idx)
        elif group == 'all':
            tmp = self.datasets[name]
        else:
            raise Exception("Group '{}' not recognized".format(group))
        return tmp

    def __subset_by_group(self, name, idx):
        """
        Method for creating subset of self.datasets[name] where items
        in idx are included.

        name (str): Name of subset to be subsetted (e.g. 'testing_set')
        idx (np.array([bool])): Boolean array for items to be included.

        Notes:
        -This function creates new arrays.
        -Storing these would double the need for memory.
        """
        X = self.datasets[name]['X'][idx, :]
        y = self.datasets[name]['y'][idx]
        t = self.datasets[name]['t'][idx]
        z = self.datasets[name]['z'][idx]
        # Note that 'r' in data filtered by treatment group is non-sensical.
        return {'X': X, 'y': y, 't': t, 'z': z}


class DatasetWrapper(Dataset):
    """
    Class for wrapping datasets from class above into format accepted
    by torch.utils.data.Dataloader.
    """
    def __init__(self, data):
        """
        Args:
        data (dict): Dictionary with 'X', 'y', 'z', 't', and in
         some cases 'r'.
        """
        self.data = data

    def __len__(self):
        return self.data['X'].shape[0]

    def __getitem__(self, idx):
        # Pytorch needs floats.
        X = self.data['X'][idx, :]
        y = self.data['y'][idx].astype(np.float64)
        z = self.data['z'][idx].astype(np.float64)
        t = self.data['t'][idx].astype(np.float64)
        if 'r' in self.data.keys():
            r = self.data['r'][idx].astype(np.float64)
            return {'X': X, 'y': y, 'z': z, 't': t, 'r': r}
        else:
            # Datasets don't have 'r' after filtering by group.
            # ('r' would be non-sensical in that case)
            return {'X': X, 'y': y, 'z': z, 't': t}


# Get some data quickly:
def get_criteo_test_data():
    data = DatasetCollection("./datasets/criteo_100k.csv", DATA_FORMAT)
    return data

def get_hillstrom_data():
    data = DatasetCollection("./datasets/" + HILLSTROM_FORMAT_1['file_name'],
                             HILLSTROM_FORMAT_1)
    return data
