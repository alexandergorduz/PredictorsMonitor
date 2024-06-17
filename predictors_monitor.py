import numpy as np
import pandas as pd
from typing import Dict



class PredictorsMonitor:
    """
    Class PredictorsMonitor for creating predictors monitor object.
    """


    def __init__(self, bins_amt: int = 10) -> None:
        """
        Initialize the PredictorsMonitor class.

        Args:
            bins_amt (int): bins amount for numerical predictors.
        """

        self.bins_amt = bins_amt
        self.etalon_stat = None


    def fit(self, df: pd.DataFrame, checks: Dict = None) -> None:
        """
        Performs a fitting etalon data.

        Args:
            df (pd.DataFrame): etalon DataFrame with predictors.
            checks (Dict): custom checks for predictors.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input data should be a Pandas DataFrame instance."
            )
        
        self.preds = [pred for pred in df.columns if df[pred].dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64', 'object', 'category']]

        if len(self.preds) == 0:
            raise IndexError(
                "There are no suitable predictors."
            )
        
        self.preds_types = {pred: ("NUMERICAL" if df[pred].dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64'] else "CATEGORY") for pred in self.preds}

        self.checks = Dict.fromkeys([f"{pred}__PSI" for pred in self.preds], 0.2)
        self.checks.update(Dict.fromkeys([f"{pred}__NA_PERC" for pred in self.preds], 0.1))

        if checks is not None:

            if not isinstance(checks, Dict):
                raise TypeError(
                    "Checks should be a Dict instance."
                )
            
            if not all(check in self.checks for check in checks):
                raise IndexError(
                    "Some incomed checks unacceptable."
                )
            
            self.checks.update(checks)
        
        self.etalon_stat = {}

        for (pred, sr) in df.items():

            pred_stat = {}

            av_amt = sr.notna().sum()
            na_amt = sr.isna().sum()

            pred_stat['NA_PERC'] = round(na_amt / (av_amt + na_amt), 5)

            pred_stat['BINS'] = {}

            if sr.dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:

                bins = np.linspace(0.0, 1.0, self.bins_amt + 1)
                bins = sr.quantile(bins).values
                bins = np.unique(bins)
                bins = np.column_stack([bins[:-1], bins[1:]]).tolist()

                pred_stat['BINS_RANGES'] = bins

                for i, (left, right) in enumerate(bins):

                    if i == 0:
                        b_num = round(sr[sr <= right].shape[0] / av_amt, 5)
                    elif i == len(bins) - 1:
                        b_num = round(sr[left < sr].shape[0] / av_amt, 5)
                    else:
                        b_num = round(sr[(left < sr) & (sr <= right)].shape[0] / av_amt, 5)
                    
                    pred_stat['BINS'][f'B_{i + 1}'] = b_num
            
            else:

                bins = sr.unique()

                for binn in bins:

                    if pd.isna(binn):
                        b_num = round(na_amt / (av_amt + na_amt), 5)
                    else:
                        b_num = round(sr[sr == binn].shape[0] / (av_amt + na_amt), 5)
                    
                    pred_stat['BINS'][f'{binn}'] = b_num
            
            self.etalon_stat[pred] = pred_stat
    

    def monitor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a monitoring test data.

        Args:
            df (pd.DataFrame): test DataFrame with predictors to monitor.
        
        Returns:
            pd.DataFrame: summary monitoring information DataFrame.
        """

        return self._check_stats(self.etalon_stat, self.get_test_stat(df[self.preds]))


    def get_test_stat(self, df: pd.DataFrame) -> Dict:
        """
        Performs a summary dictionary of test data in perspective of etalon data.

        Args:
            df (pd.DataFrame): test DataFrame with predictors to monitor.
        
        Returns:
            Dict: summary dictionary of test data in perspective of etalon data.
        """

        if self.etalon_stat is None:
            raise Exception(
                "PredictorsMonitor is not fitted yet."
            )
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input data should be a Pandas DataFrame instance."
            )
        
        if not all(pred in df.columns for pred in self.preds):
            raise IndexError(
                "Different predictors between etalon and test DataFrame's."
            )
        
        test_stat = {}

        for (pred, sr) in df.items():

            pred_stat = {}

            av_amt = sr.notna().sum()
            na_amt = sr.isna().sum()

            pred_stat['NA_PERC'] = round(na_amt / (av_amt + na_amt), 5)

            pred_stat['BINS'] = {}

            if sr.dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:

                bins = self.etalon_stat[pred]['BINS_RANGES']

                for i, (left, right) in enumerate(bins):

                    if i == 0:
                        b_num = round(sr[sr <= right].shape[0] / av_amt, 5)
                    elif i == len(bins) - 1:
                        b_num = round(sr[left < sr].shape[0] / av_amt, 5)
                    else:
                        b_num = round(sr[(left < sr) & (sr <= right)].shape[0] / av_amt, 5)
                    
                    pred_stat['BINS'][f'B_{i + 1}'] = b_num
            
            else:

                bins = sr.unique()

                for binn in bins:

                    if pd.isna(binn):
                        b_num = round(na_amt / (av_amt + na_amt), 5)
                    else:
                        b_num = round(sr[sr == binn].shape[0] / (av_amt + na_amt), 5)
                    
                    pred_stat['BINS'][f'{binn}'] = b_num
            
            test_stat[pred] = pred_stat
        
        return test_stat
    

    def _check_stats(self, etalon: Dict, test: Dict) -> pd.DataFrame:
        """
        Performs a creating summary monitoring information DataFrame.

        Args:
            etalon (Dict): summary dictionary of etalon data.
            test (Dict): summary dictionary of test data in perspective of etalon data.
        
        Returns:
            pd.DataFrame: summary monitoring information DataFrame.
        """

        res = pd.DataFrame(columns=['PRED_NAME', 'PRED_TYPE', 'CHECK_TYPE', 'CHECK_VALUE', 'CHECK_STATE'])

        for pred in self.preds:

            pred_res = []

            row_tmplt = {'PRED_NAME': pred, 'PRED_TYPE': self.preds_types[pred], 'CHECK_STATE': 'OK'}

            etalon_bins = set(etalon[pred]['BINS'].keys())
            test_bins = set(test[pred]['BINS'].keys())
            inter_bins = list(etalon_bins.intersection(test_bins))

            if len(inter_bins) == 0:

                psi = None
            
            else:

                a = np.array([etalon[pred]['BINS'][binn] + 1e-10 for binn in inter_bins])
                b = np.array([test[pred]['BINS'][binn] + 1e-10 for binn in inter_bins])
                psi = round(np.sum((a - b) * np.log(a / b)), 5)
            
            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'PSI'
            row['CHECK_VALUE'] = f'{psi}'
            if psi is None or np.isnan(psi) or psi > self.checks[f'{pred}__PSI']:
                row['CHECK_STATE'] = 'NOT_OK'
            pred_res.append(row)

            etalon_na_perc = etalon[pred]['NA_PERC']
            test_na_perc = test[pred]['NA_PERC']

            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'NA_PERC'
            row['CHECK_VALUE'] = f'{test_na_perc}'
            if abs(etalon_na_perc - test_na_perc) > self.checks[f'{pred}__NA_PERC']:
                row['CHECK_STATE'] = 'NOT_OK'
            pred_res.append(row)

            if self.preds_types[pred] == "CATEGORY":

                row = row_tmplt.copy()
                row['CHECK_TYPE'] = 'NEW_VAL'
                if test_bins - etalon_bins:
                    row['CHECK_VALUE'] = f'{list(test_bins - etalon_bins)}'.replace('[', '| ').replace(']', ' |').replace(', ', ' | ')
                    row['CHECK_STATE'] = 'NOT_OK'
                else:
                    row['CHECK_VALUE'] = f'{None}'
                pred_res.append(row)

                row = row_tmplt.copy()
                row['CHECK_TYPE'] = 'NO_VAL'
                if etalon_bins - test_bins:
                    row['CHECK_VALUE'] = f'{list(etalon_bins - test_bins)}'.replace('[', '| ').replace(']', ' |').replace(', ', ' | ')
                    row['CHECK_STATE'] = 'NOT_OK'
                else:
                    row['CHECK_VALUE'] = f'{None}'
                pred_res.append(row)
            
            res = pd.concat([res, pd.DataFrame(pred_res)], ignore_index=True)
        
        return res