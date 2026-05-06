import numpy as np
import pandas as pd
from RK4Sim import *
from EulerSim import *

data = pd.read_parquet(f'../../../generator/src/output/flight_{0}.parquet')
thrust = dict(np.genfromtxt("../../../source_model/R7_SIMLE/R7_OUTPUT/thrust_source.csv", delimiter=","))


start_mass = 63.43882806087689 # todo policzyc na podstawie wykresow
start_position = np.array([0, 0, 10])
time = 11
angle = np.array([0, 0, 1])

new_position, new_velocity, new_acceleration = euler_t(start_position, start_mass, angle, time, thrust)