"""
Conversion rate experiments for uplift modeling. In this file,
the experiments run investigate the effect of baseline conversion
rate in the data vs. performance of undersampling. The experiments
are done on the "visit" label in the Criteo dataset.

Replace the contents of "files" to match the files you have created.
"""

# Enable imports from parent directory:
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import gzip
import numpy as np
import pickle
import load_data
import pickle_dataset
import cvt_underclass_test
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

# path to dataset files
data_path = './synthetic_data/'
path = "./"
# dataset files to iterate over
# Keep conversion rate order of files!!!
# files = ['criteo-uplift.csv0.001.3756116055.gz', 
#         'criteo-uplift.csv0.002.1041642162.gz',
#         'criteo-uplift.csv0.005.1345760781.gz',
#         'criteo-uplift.csv0.01.125476986.gz',
#         'criteo-uplift.csv0.02.2992998466.gz', 
#         'criteo-uplift.csv0.041.2587620614.gz']
# corresponding_conversion_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.041]

def run_experiment():
    results = []
    files = []
    conversion_rates = []
    for file_ in listdir(data_path):
        file_location = join(data_path, file_)
        if not isfile(file_location):
            # If item is not a file, loop to next
            continue
        if not file_.startswith('criteo-uplift.csv0'):
            continue
        if not file_.endswith('.gz'):
            continue
        # Add file to list of files to process:
        files.append(file_)
        # Extract conversion rate from file name:
        tmp = file_[17:22]
        if tmp.endswith('.'):
            tmp = tmp[:-1]
        conversion_rates.append(tmp)
        
    for file_, conversion_rate in zip(files, conversion_rates):
        # Experiments!
        # Load data. All training, validation, and testing sets
        # are ready.
        print("Reading in data from {}...".format(file_))
        tmp_data = pickle_dataset.load_pickle(file_, data_path)
        # Find optimal k for validation set.
        # What's the range of k's to be tested for different conversion rates?
        # Range should be 1 to 50% with some reasonable step sizes?
        i_min = 0
        i_max = int(.5 / conversion_rate)
        steps = 20  # Is this okish? Makes step size 1 for conversion rate .041.
        step_size = max(1, int((i_max - i_min) / steps)) # Step size at least 1.
        # Do we perhaps care how this evolves over k? Store all, figure out later?
        # Estimate metrics (mostly AUUC) for optimal k with CVT-LR
        # Perhaps also train a CVT-LR model for baseline as reference?
        print("Training model...")
        tmp_results = cvt_underclass_test.cvt_experiment(tmp_data,
                                                         k_min=i_min,
                                                         k_max=i_max,
                                                         k_step=step_size)
        print("Saving results...")
        results.append([conversion_rate, tmp_results])  # Does this include optimal k?
        #with gzip.open(path +"results_for_" + str(conversion_rate) + ".gz", "wb") as handle:
        with gzip.open(path + "results_for_" + file_ + ".gz", "wb") as handle:
            pickle.dump(tmp_results, handle)
        # Store results in list with details on k and conversion rate in data (?)
    # Store result list
    print("Storing all results in all_results.gz")
    with gzip.open(path + "all_results.gz", "wb") as handle:
        pickle.dump(results, handle)
    # Get back to it for plotting.
    

    
if __name__ == '__main__':
    # Run main program
    run_experiment()
