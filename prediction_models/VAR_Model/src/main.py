import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

from var import create_var
from vecm import create_vecm
 
def read_sensor_data(directory, index='Time'):
    sensors = pd.read_csv(directory)
    sensors[index] = pd.to_datetime(sensors[index])
    sensors.set_index(index, inplace=True)
    return sensors

def check_final_diff_order(data, signif = 0.05, test_sample = 20000, max_diff_order = 2):
    # checking how many times each of the columns need 
    # to be differenced for them to become stationary
    # and choosing the biggest value 
    final_diff_order=0
    for col in data.columns:
        # pereparing the data for the ADF test - using only a small part of our data
        # since the test will not work for datasets as large as those generated in our project
        series = data[col].dropna().iloc[:test_sample]

        # ensuring there are no constant columns - otherwise adfuller will crash
        if series.std() == 0:
            print(f"Skipping constant column {col}.")
            continue
        
        current_diff_order = 0
        # testing stationarity using the ADF test 
        # the column is stationary if the  p value calculated in the test
        # and stored in results[1] is less or equal 0.05 (signif)
        p_value = adfuller(series, autolag="AIC")[1]

        # differencing the column until it becomes stationary or
        # until the data is differenced maximum number of times
        while p_value > signif and current_diff_order < max_diff_order:
            series = series.diff().dropna()
            p_value = adfuller(series, autolag="AIC")[1]
            current_diff_order += 1

        # tracking the highest differencing order needed across all columns
        final_diff_order = max(final_diff_order, current_diff_order)
        
        if p_value > signif:
            print(f"Warning: Column {col} is still not stationary after {max_diff_order} differences.")
    return final_diff_order

def is_cointegrated(data, col_idx = 1, det_order = -1, max_lag = 20):
    # Calculating the number of lagged differences in the model (k_arr_diff)
    # TODO: when a better approach to calculate lag values is implemented
    #       we can use those lag values here instead of recalculating them
    res = VAR(data).select_order(maxlags=max_lag)

    k_ar_diff = res.selected_orders.get("aic")
    if k_ar_diff is None:
       k_ar_diff = 1

    # Performing the Johanson's Cointegration Test
    # det_order controls deterministic terms 
    # (here det_order = -1 so there are no deterministic terms)
    result = coint_johansen(data, det_order, k_ar_diff)

    # Extracting trace test statistics to measure
    # the strength of cointegration relationships
    trace = result.lr1
    
    # Extracting  critical values for significance level
    # chosen by col_idx (here we have 1 so it's 5%)
    # critical values are a boundary from which we see the output
    # as statistcally significant
    crit = result.cvt[:, col_idx]

    # Comparing test statistics with critical values
    # If trace statistic is greater than the critical value there is evidence of cointegration
    return sum(t > c for t, c in zip(trace, crit)) > 0

def choose_model(data, final_diff_order):
    # if the data is already stationary we use VAR
    # otherwise additional checks are needed 
    if final_diff_order > 0:
        # VAR will not work properly on cointegrated data, therefore
        # we need to do a cointegration test and use a different model
        # if the data is cointegrated
        if is_cointegrated(data):
            print("The series is cointegrated. The VECM model will be used.")
            return "VECM"
        else:
            print("The series is not cointegrated. The VAR model will be used.")
            print(f"The series was differenced {final_diff_order} times.")
            return "VAR"
    else:
        print("The series was already stationary. The VAR model will be used.")
        return "VAR"    

def prepare_data(data, model, final_diff_order):
    # preprocessing the data - if it isn't stationary but it also isn't cointegrated
    # (ie if we use the VAR model for non-stationary data) the whole time series has to be
    # differenced the maximum amount of times needed for a single column to become stationary
    # for a VAR model with data that was already stationary and the VECM model no
    # additional preprocessing is needed
    new_data = data.copy()
    if (model=="VAR" and final_diff_order > 0):
        for _ in range(final_diff_order):
            new_data = new_data.diff()
        return new_data.dropna()
    else:
        return new_data

# Reading the data from csv files
# TODO: split the data into a testing and training sets and
#       check the preferable dataset size for VAR models -
#       if the data for one flight won't be enough, look into how
#       can the data from more flights be merged witout messing up
#       the time series

training_sensors = read_sensor_data('../../../model_translator/src/output/flight_0_best_sensors.csv')

# calculating how many times the data needs to be differenced
# in order for it to become stationary
final_diff_order = check_final_diff_order(training_sensors)

# choosing the right model for the dataset (VAR or VECM)
model_type = choose_model(training_sensors, final_diff_order)

data = prepare_data(training_sensors, model_type, final_diff_order)

# TODO: calculate lag order here

if model_type=="VAR":
    model = create_var(data)
else:
    model = create_vecm(data)
