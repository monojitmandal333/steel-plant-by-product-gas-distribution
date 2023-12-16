from steel_plant_by_product_gas_distribution.data_preprocessing import *
from steel_plant_by_product_gas_distribution.LSSVM import *
from steel_plant_by_product_gas_distribution.LSTM import *
import datetime as dt
from steel_plant_by_product_gas_distribution.ensemble import *
from steel_plant_by_product_gas_distribution.LP import *


from typing import Dict

class onlineModule:
    
    def __init__(
        self,
        ensemble_model:EnsembleModel,
        lp_model:LP
    ):
        self.__em = ensemble_model
        self.__lpm = lp_model
        
    def output(
        self,
        data:pd.DataFrame
    ) -> Dict[str,float]:
        prediction = self.__em.predict(data)
        bfg_gen =  prediction[dd.bfg_gen]
        cog_gen = prediction[dd.cog_gen]
        bfs_demand = prediction[dd.bfs]
        sp_lcp_hsm_demand = prediction[dd.sp_lcp_hsm]
        h_t_minus_1 = data[dd.holder_level].values[-1]

        result = self.__lpm.find_decision_variable_output(bfg_gen, cog_gen, bfs_demand, sp_lcp_hsm_demand, h_t_minus_1)
        # Final results
        return result

# if __name__ == "__main__":
#     cv_bfg = 930
#     cv_cog = 3900
#     cv_ldg = 1600
#     cv_mxg1 = 1050
#     cv_mxg2 = 2300

#     byproduct_demand_columns = [dd.bfg_gen,dd.cog_gen,dd.bfs,dd.sp_lcp_hsm]
#     # Data Preprocessing-------------------------------------------------------------------
#     dp = DataPreprocessing(
#         k = 1.161*10**(-6),
#         hp_filter_lambda= 10,
#         n_lag = 5,
#         n_sequence=1,
#         time_aggregate_min=15
#     )
#     data = pd.read_csv("data/Production_Use_Case_Data.csv")
#     data[dd.timestamp] = pd.to_datetime(
#         data[dd.timestamp],
#         format = '%m/%d/%Y %H:%M'
#     )
#     data = data.sort_values(by = dd.timestamp)
#     # Byproduct gas demand calculation
#     data = dp.calculate_by_product_gas_demand(data)
#     # Train test split
#     df_train,df_test = dp.train_test_split(data)

#     # Outlier Correction on train data to supply good data into model
#     for col in byproduct_demand_columns:
#         df_train = dp.correct_outliers(
#             data = data,
#             column_name=col
#         )
#     # Hodrick Presscott filter on data to segregate trend and volatility in data
#     for col in byproduct_demand_columns:
#         df_train = dp.apply_hodrick_prescott_filter(
#             data = df_train,
#             column_name=col
#         )

#     # Model Training
#     ensemble_model = EnsembleModel(byproduct_demand_columns,dp)
#     lstm_models = ensemble_model.train_lstm_models(data=df_train)
#     lssvm_models = ensemble_model.train_lssvm_models(data = df_train)

#     lp = LP(
#         cv_bfg = cv_bfg,
#         cv_cog = cv_cog,
#         cv_ldg = cv_ldg,
#         cv_mxg1=cv_mxg1,
#         cv_mxg2=cv_mxg2
#     )
#     om = onlineModule(
#         ensemble_model=ensemble_model,
#         lp_model = lp)

#     for timestamp in df_test.head(100)[dd.timestamp]:
#         data = dp.get_online_record(
#             data = df_test,
#             timestamp = timestamp
#         )
#         if len(data) == 5:
#             print(om.output(data = data))