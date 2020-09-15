"""
Code for serializing datasets (which takes very long and a lot of memory)
into pickle-format. The point is to do this once, and then let other
tests use much less memory and time.

Run this once for all your experiments (unless you want to randomize the
data again).
"""
# Enable imports from parent directory:
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import gzip
import pickle
import random
import load_data


CRITEO_FORMAT = load_data.DATA_FORMAT
# CRITEO_FORMAT['file_name'] = 'criteo_100k.csv'  # This is for testing.
CRITEO_FORMAT['file_name'] = 'criteo-uplift.csv'  # This is the actual experiment.
CRITEO_FORMAT['y_idx'] = 14  # for the "visit" label. Using 'conversion' (13) by default.


# HILLSTROM_FORMAT = load_data.HILLSTROM_FORMAT_1
# Already contains filename.

#DATASETS = [HILLSTROM_FORMAT, CRITEO_FORMAT]
DATASETS = [CRITEO_FORMAT]

def create_pickles(path='../datasets/', set_conversion_rate=None):
    """
    Function for pickling datasets.

    Args:
    path (str): Path of datasets. Actual file names are
     specified in DATASETS-list above.
    """
    for dataset in DATASETS:
        seed = random.randrange(2**32 - 1)
        dataset['random_seed'] = seed
        # Load data and do preprocessing:
        data = load_data.DatasetCollection(path + dataset['file_name'],
                                           dataset,
                                           set_conversion_rate=set_conversion_rate)
        # Write as pickle to disk:
        new_path = "./synthetic_data/"
        with gzip.open(new_path + dataset['file_name'] + str(set_conversion_rate) + "." + str(seed) + '.gz',
                  'wb') as handle:
            pickle.dump(data, handle)


def load_pickle(file_name, path='./datasets/'):
    """
    Function for loading pickled datasets. Zipped or unzipped.

    Args:
    file_name (str): Name of file
    path (str): Location of file
    """
    from load_data import DatasetCollection
    # Do gzip version
    fp = gzip.open(path + file_name, 'rb') if file_name.endswith(".gz") else open(path + file_name, 'rb')
    data = pickle.load(fp)
    fp.close()
    #with open(path + file_name, 'rb') as handle:
    #    data = pickle.load(handle)
    return data


if __name__ == '__main__':
    # If main program,
    # create pickle-files.
    for rate in [.041, .02, .01, .005, .002, .001]:
        create_pickles(set_conversion_rate = rate)
