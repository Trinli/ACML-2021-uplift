# Undersampling and calibration for uplift modeling
Code for conversion rate experiments for uplift modeling.

1. Run the experiments in the parent directory. Following those instructions,
   you will get data into appropriate folders used in this experiment.
2. Run 
   ```python pickle_dataset.py´´´.
   This will create synthetic datasets with
   conversion rates varying from 0.001 to 0.041.
3. Run 
   ```python conversion_rate_experiments.py´´´.
   This will run the actual
   experiments. The results are printed to screen, although they are also
   stored in a rather convoluted format to file. We will have to fix that
   before publishing.
