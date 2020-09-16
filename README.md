# Undersampling and calibration for uplift modeling

Code used to produce results presented in the publication "Undersampling and Calibration for Uplift Modeling".


## Code preparation
The code was tested using Python 3. 
Requirements are listed in requirements.txt.


## Data preparation
1. Download data
 - http://ailab.criteo.com/criteo-uplift-prediction-dataset/
 - https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
2. Extract data to .csv-file and store in ./datasets/
3. Run pickle_dataset.py to perform the normalization we used in the experiments.
   The script also splits the data randomly into training, validation, and testing sets,
   and creates a new label by running the class-variable transformation.
   The dataset can be zipped (.gz) to save space, but this is not mandatory.
   We recommend reserving 120 GB of RAM for this.
   In the paper, we ran this 10 times to get 10 differently ranomized data sets.
   Be patient. Expect this step to take hours.

## Experiments
1. Run undersampling experiments by running undersampling_experiments.py with suitable
   parameters, e.g.:
   ```python undersampling_experiments.py ./datasets/criteo-uplift.csv123.pickle.gz cvt 1,600,10```
    (replace '123' with whatever your file is named, 'cvt' refers to class-variable
    transformation, '1,600,10' indicates "test k from 1 to 600 with a step of 10").
    Note that the last print section shows the testing set metrics for the best model.
2. Run isotonic regression experiments, e.g.:
   ```python isotonic_regression_experiments.py ./dataset/criteo-uplift.csv123.pickle.gz dclr 3```
   (replace '123' with your dataset file, 'dclr' refers to double-classifier with
   logistic regression, '3' refers to k=3).
3. Results are printed to screen and stored in uplift_results.csv. Look for rows with 
   'Test description' set to 'testing set'.

The alternative models for both undersampling and isotonic regression experiments are
 - 'dc' (or 'dclr'): double-classifier with logistic regression
 - 'dcrf': double-classifier with random forest
 - 'cvt' (or 'cvtlr'): class-variable transformation with logistic regression
 - 'cvtrf': class-variable transformation with random forest

In the paper, we created 10 randomized data sets, ran the code 10 times and averaged the results.
For visualizations, use function plot_uplift_curve() in uplift_metrics.py.

