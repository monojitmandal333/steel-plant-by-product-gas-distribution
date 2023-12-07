from data_preprocessing import *
from LSSVM import *
from LSTM import *
import datetime as dt
from ensemble import *

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

for timestamp in df_test.head(100)[dd.timestamp]:
    data = dp.get_online_record(
        data = df_test,
        timestamp = timestamp
    )
    if len(data) == 5:
        print(ensemble_model.predict(data = data))