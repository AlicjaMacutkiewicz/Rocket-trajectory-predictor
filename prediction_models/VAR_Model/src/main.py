import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

flight0_sensors = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
flight0_data = pd.read_csv('../../../model_translator/src/output/flight_0.out', header=None)

flight1_sensors = pd.read_csv('../../../model_translator/src/output/flight_1_best_sensors.csv')
flight1_data = pd.read_csv('../../../model_translator/src/output/flight_1.out', header=None)

flight2_sensors = pd.read_csv('../../../model_translator/src/output/flight_2_best_sensors.csv')
flight2_data = pd.read_csv('../../../model_translator/src/output/flight_2.out', header=None)

