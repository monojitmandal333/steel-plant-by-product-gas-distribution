import numpy as np
import pandas as pd
import statsmodels.api as sm
import steel_plant_by_product_gas_distribution.data_dictionary as dd
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
"""Data Preprocessing Module"""


class DataPreprocessing():
    """Data Import, HP filter, Convert into Supervised Data"""
    def __init__(
        self,
        k:float,
        hp_filter_lambda:float,
        n_lag:int,
        n_sequence:int,
        time_aggregate_min:int
    ):
        self.__k = k # kcal/hr to MW conversion factor
        self.__hp_filter_lambda = hp_filter_lambda
        self.__n_lag = n_lag
        self.__n_seq = n_sequence
        self.__column_min_max = {}
        self.__time_agg_min = time_aggregate_min
    
    def calculate_by_product_gas_demand(
        self,
        data:pd.DataFrame
    ) -> pd.DataFrame:
        data[dd.bfs] = (data[dd.bfg_to_stove]*data[dd.bfg_cv] + data[dd.cog_to_bf_stove]*data[dd.cog_cv])*self.__k
        data[dd.sp_lcp_hsm] = (
            (data[dd.cog_from_gmbs_2] + data[dd.cog_from_gmbs_3] + 
             data[dd.cog_from_gmbs_4] + data[dd.cog_from_gmbs_5])*data[dd.cog_cv]
            + (data[dd.bfg_from_gmbs_2] + data[dd.bfg_from_gmbs_3] + 
               data[dd.bfg_from_gmbs_4] + data[dd.bfg_from_gmbs_5])*data[dd.bfg_cv])*self.__k
        return data
    
    def train_test_split(
        self,
        data:pd.DataFrame
    ) -> pd.DataFrame:
        tscv = TimeSeriesSplit()
        for i,(train_ix,test_ix) in enumerate(tscv.split(data)):
            if i == 4:
                df_train = data[data.index.isin(train_ix)]
                df_test = data[data.index.isin(test_ix)]
        return (df_train,df_test)
    
    def correct_outliers(self,data,column_name):
        iqr = data[column_name].quantile(0.75) - data[column_name].quantile(0.25)
        lcl = data[column_name].quantile(0.25) - 1.5*iqr
        ucl = data[column_name].quantile(0.75) + 1.5*iqr
        data[column_name][data[column_name] < lcl] = lcl
        data[column_name][data[column_name] > ucl] = ucl
        return data

    def apply_hodrick_prescott_filter(self,data,column_name):
        data[f"{column_name}_vol"],data[f"{column_name}_trend"] = sm.tsa.filters.hpfilter(
            data[column_name],
            lamb=self.__hp_filter_lambda
        )
        return data
    
    def scale_feature(self,data,column_name):
        if column_name in self.__column_min_max.keys():
            min = self.__column_min_max[column_name]["min"]
            max = self.__column_min_max[column_name]["max"]
        else:
            min = data[column_name].min()
            max = data[column_name].max()
            self.__column_min_max[column_name] = {
                "min":data[column_name].min(),
                "max":data[column_name].max()
            }
        data[column_name] = (data[column_name] - min)/(max - min)
        return data
    
    def inverse_scale_feature(self,column_value,column_name):
        min = self.__column_min_max[column_name]["min"]
        max = self.__column_min_max[column_name]["max"]
        column_value = min + (max - min)*column_value
        return column_value
    
    def get_independent_col_values(self,data):
        cols = list()
        for i in range(self.__n_lag-1, -1, -1):
            cols.append(data.shift(i))
        return cols
    
    def get_dependent_col_values(self,data):
        cols = list()
        for i in range(1, self.__n_seq+1):
            cols.append(data.shift(-i))
        return cols
    
    def get_independent_col_names(self,column_name):
        col_names = list()
        for i in range(self.__n_lag-1, -1, -1):
            if i == 0:
                col_names.append(f'{column_name}(t)')
            else:
                col_names.append(f'{column_name}(t-{i})')
        return col_names
    
    def get_dependent_col_names(self,column_name):
        col_names = list()
        for i in range(1, self.__n_seq+1):
            col_names.append(f'{column_name}(t+{i})')
        return col_names
    
    def convert_series_to_feature(
        self,
        data,
        column_name,
        dropnan = True
    ):
        data = data[[column_name]]
        # input sequence (t-n, ... t-1)
        X_list = self.get_independent_col_values(data)
        X_names = self.get_independent_col_names(column_name)
        df = pd.concat(X_list, axis=1)
        df.columns = X_names
        if dropnan:
            df = df.dropna()
        return df

    def convert_series_to_supervised(
        self,
        data,
        column_name,
        dropnan = True
    ):
        data = data[[column_name]]
        # input sequence (t-n, ... t-1)
        X_list = self.get_independent_col_values(data)
        X_names = self.get_independent_col_names(column_name)
        # forecast sequence (t, t+1, ... t+n)
        y_list = self.get_dependent_col_values(data)
        y_names = self.get_dependent_col_names(column_name)

        values = X_list + y_list
        columns = X_names + y_names

        # put it all together
        df = pd.concat(values, axis=1)
        df.columns = columns
        # drop rows with NaN values
        if dropnan:
            df.dropna(inplace = True)
        return df
    
    def split_X_y(self,data,column_name):
        X_columns = self.get_independent_col_names(column_name)
        y_columns = self.get_dependent_col_names(column_name)
        X = np.array(data[X_columns])
        y = np.array(data[y_columns])
        return (X,y)
    
    def get_online_record(self,data,timestamp):
        data = data[
            (data[dd.timestamp] <= timestamp) & 
            (data[dd.timestamp] >= timestamp - dt.timedelta(minutes=self.__time_agg_min*(self.__n_lag-1)))
        ].copy()
        return data

