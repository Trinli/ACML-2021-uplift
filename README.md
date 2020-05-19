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
   The dataset can also be zipped (.gz) to save space, but this is not mandatory.
5. Run undersampling experiments by running undersampling_experiments.py with suitable
   parameters, e.g.:
   python undersampling_experiments.py ./datasets/criteo-uplift.csv123.pickle.gz cvt 1,600,10
    (replace '123' with whatever the your file is named).
    Note that the last print section shows the testing set metrics for the best model
    for your testing set.
6. Run isotonic regression experiments.
   -Tomasz, details please?

# CIKM-2020-uplift
