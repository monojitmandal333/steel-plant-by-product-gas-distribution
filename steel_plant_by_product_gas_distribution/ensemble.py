import steel_plant_by_product_gas_distribution.data_dictionary as dd
from steel_plant_by_product_gas_distribution.LSTM import *
from steel_plant_by_product_gas_distribution.LSSVM import *
from steel_plant_by_product_gas_distribution.data_preprocessing import *
from typing import List,Dict

class EnsembleModel():
    def __init__(
        self,
        byproduct_gas_demand_columns:List[str],
        data_preprocessing:DataPreprocessing
    ):
        self.byproduct_gas_demand_columns = byproduct_gas_demand_columns
        self.__lstm_models = None
        self.__lssvm_models = None
        self.__dp = data_preprocessing

    def train_lstm_models(
        self,
        data:pd.DataFrame
    ) -> List[LSTMTrend]:
        lstm_models = {}
        for col in self.byproduct_gas_demand_columns:
            col_trend = f"{col}_trend"
            df_train = self.__dp.scale_feature(data=data,column_name=col_trend)
            df_train = self.__dp.convert_series_to_supervised(
                data = df_train,
                column_name = col_trend,
                dropnan = True
            )
            print(f"LSTM model training for {col_trend}")
            X_train,y_train = self.__dp.split_X_y(
                data = df_train,
                column_name = col_trend
            )
            lstm = LSTMTrend(
                n_epoch = 30,
                n_batch = 10,
                n_neurons = 5
            )
            lstm.fit(X_train,y_train)
            lstm_models[col_trend] = lstm
        self.__lstm_models = lstm_models
        return lstm_models
    
    def train_lssvm_models(
        self,
        data:pd.DataFrame
    ) -> List[LSSVMVolatility]:
        lssvm_models = {}
        for col in self.byproduct_gas_demand_columns:
            col_vol = f"{col}_vol"
            df_train = self.__dp.scale_feature(data=data,column_name=col_vol)
            df_train = self.__dp.convert_series_to_supervised(
                data = df_train,
                column_name = col_vol,
                dropnan = True
            )
            print(f"LSSVM model training for {col_vol}")
            X_train,y_train = self.__dp.split_X_y(
                data = df_train.sample(5000),
                column_name = col_vol
            )
            lssvm = LSSVMVolatility()
            lssvm.fit(X_train,y_train)
            lssvm_models[col_vol] = lssvm
        self.__lssvm_models = lssvm_models
        return lssvm_models
    
    def predict(
        self,
        data:pd.DataFrame
    ) -> Dict[str,float]:
        predict_dict = {}
        for col in self.byproduct_gas_demand_columns:
            data = self.__dp.apply_hodrick_prescott_filter(
                data = data,
                column_name = col
            )
            col_trend = f"{col}_trend"
            col_vol = f"{col}_vol"
            df_trend = self.__dp.scale_feature(
                data = data,
                column_name = col_trend
            )
            df_trend = self.__dp.convert_series_to_feature(
                data = df_trend,
                column_name = col_trend,
                dropnan = True
            )
            df_vol = self.__dp.scale_feature(
                data = data,
                column_name = col_vol
            )
            df_vol = self.__dp.convert_series_to_feature(
                data = df_vol,
                column_name = col_vol,
                dropnan = True
            )
            X_test = np.array(df_trend)
            model = self.__lstm_models[col_trend]
            y_pred_trend = model.predict(X_test)
            y_pred_trend = self.__dp.inverse_scale_feature(y_pred_trend[0][0],col_trend)

            X_test = np.array(df_vol)
            model = self.__lssvm_models[col_vol]
            y_pred_vol = model.predict(X_test)
            y_pred_vol = self.__dp.inverse_scale_feature(y_pred_vol[0],col_vol)

            predict_dict[col_trend] = y_pred_trend
            predict_dict[col_vol] = y_pred_vol
            predict_dict[col] = y_pred_trend + y_pred_vol
        return predict_dict
