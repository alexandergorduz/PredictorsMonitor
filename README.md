## PredictorsMonitor Class

The `PredictorsMonitor` class is designed for monitoring and comparing statistical characteristics of predictors between an "etalon" dataset (used for training) and a test dataset. It supports both numerical and categorical predictors, providing insights into data drift or discrepancies that might affect model performance or data quality.

### Features

1. **Initialization (`__init__`)**:
   - `bins_amt`: Specifies the number of bins for numerical predictors.
   - Initializes with `etalon_stat` set to `None`.

2. **Fitting (`fit`)**:
   - Accepts a Pandas DataFrame (`df`) as the etalon dataset and performs initial setup.
   - Identifies suitable predictors based on their data types (numerical or categorical).
   - Computes statistical metrics such as missing value percentages and bin distributions for each predictor.
   - Allows customizing checks to monitor predictor drift or changes (`checks` parameter).

3. **Monitoring (`monitor`)**:
   - Evaluates a test dataset to compare statistical metrics against the etalon dataset.
   - Reports discrepancies or drifts in terms of Predictive Stability Index (PSI), missing value percentages, and changes in categorical values.

4. **Utility Methods**:
   - `get_test_stat`: Computes summary statistics for a test dataset.
   - `_check_stats`: Compares statistics between the etalon and test datasets, generating a detailed report in a DataFrame format.