# Undersampling and calibration for uplift modeling
Code for undersampling and calibration for uplift
modeling relating to the publication ... blah, blah.

1. Download data
-http://ailab.criteo.com/criteo-uplift-prediction-dataset/
-https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
2. Install python requirements listed in requirements.txt (some might be unnecessary).
3. Extract data to .csv-file and store in ./datasets/
4. Run pickle_dataset.py. This does the normalization we used in the experiments and
   randomly splits the data into training, validation, and testing sets. This also
   does the class-variable transformation to the label.
   The dataset can be zipped (.gz) to save space, but this is not mandatory.
5. Run undersampling experiments by running undersampling_experiments.py with suitable
   parameters, e.g.:
   ```python undersampling_experiments.py ./datasets/criteo-uplift.csv123.pickle.gz cvt 1,600,10```
    (replace '123' with whatever your file is named, 'cvt' refers to class-variable
    transformation, '1,600,10' indicates "test k from 1 to 600 with a step of 10").
    Note that the last print section shows the testing set metrics for the best model.
6. Run isotonic regression experiments, e.g.:
   ```python isotonic_regression_experiments.py ./dataset/criteo-uplift.csv123.pickle.gz dclr 3```
   (replace '123' with your dataset file, 'dclr' refers to double-classifier with
   logistic regression, '3' refers to k=3.

The alternative models for both undersampling and isotonic regression experiments are
*'dc' (or 'dclr'): double-classifier with logistic regression
*'dcrf': double-classifier with random forest
*'cvt' (or 'cvtlr'): class-variable transformation with logistic regression
*'cvtrf': class-variable transformation with random forest

# CIKM-2020-uplift
