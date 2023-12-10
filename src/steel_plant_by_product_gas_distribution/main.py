from data_preprocessing import *
from LSSVM import *
from LSTM import *
import datetime as dt
from ensemble import *
from online_module import *
from enum import Enum

# By Product Gas Parameters----------------------------------------------------------
class calorificValues(Enum):
    cv_bfg = 930
    cv_cog = 3900
    cv_ldg = 1600
    cv_mxg1 = 1050
    cv_mxg2 = 2300

byproduct_demand_columns = [dd.bfg_gen,dd.cog_gen,dd.bfs,dd.sp_lcp_hsm]


# Data Preprocessing-------------------------------------------------------------------
dp = DataPreprocessing(
     k = 1.161*10**(-6),
     hp_filter_lambda= 10,
     n_lag = 5,
     n_sequence=1,
     time_aggregate_min=15
)
data = pd.read_csv("data/Production_Use_Case_Data.csv")
data[dd.timestamp] = pd.to_datetime(
    data[dd.timestamp],
    format = '%m/%d/%Y %H:%M'
)
data = data.sort_values(by = dd.timestamp)
# Byproduct gas demand calculation
data = dp.calculate_by_product_gas_demand(data)
# Train test split
df_train,df_test = dp.train_test_split(data)

# Outlier Correction on train data to supply good data into model
for col in byproduct_demand_columns:
    df_train = dp.correct_outliers(
        data = data,
        column_name=col
    )
# Hodrick Presscott filter on data to segregate trend and volatility in data
for col in byproduct_demand_columns:
    df_train = dp.apply_hodrick_prescott_filter(
        data = df_train,
        column_name=col
    )

# Model Training
ensemble_model = EnsembleModel(byproduct_demand_columns,dp)
lstm_models = ensemble_model.train_lstm_models(data=df_train)
lssvm_models = ensemble_model.train_lssvm_models(data = df_train)

lp = LP(
    cv_bfg = calorificValues.cv_bfg.value,
    cv_cog = calorificValues.cv_cog.value,
    cv_ldg = calorificValues.cv_ldg.value,
    cv_mxg1=calorificValues.cv_mxg1.value,
    cv_mxg2=calorificValues.cv_mxg2.value
)
om = onlineModule(ensemble_model,lp)

records = []
for timestamp in df_test[dd.timestamp]:
    data = dp.get_online_record(
        data = df_test,
        timestamp = timestamp
    )
    record = {"timestamp":timestamp}
    if len(data) == 5:
        prediction = ensemble_model.predict(data = data)
        for col in prediction.keys():
            record[col] = prediction[col]
        decision_variables = om.output(data = data)
        for col in decision_variables:
            record[col] = decision_variables[col]
    records.append(record)
df_test_predicted = pd.DataFrame(records)
df_test_predicted = df_test.merge(
    df_test_predicted,
    how = "left",
    left_on = "Time",
    right_on = "timestamp"
)
print(df_test_predicted)
df_test_predicted.to_csv("data/output.csv")