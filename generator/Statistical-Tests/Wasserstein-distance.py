# how much is data linear correlated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance

columns = [
    "Best_Acc_X",
    "Best_Acc_Y",
    "Best_Acc_Z",
    "Best_AngVel_X",
    "Best_AngVel_Y",
    "Best_AngVel_Z",
    "Barometer_Value",
    "Sensor_Value",
    "Thrust",
    "Mass",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Acceleration_X",
    "Acceleration_Y",
    "Acceleration_Z",
]

N = 12
matrix = np.zeros((N, N))

flights = []
for i in range(N):
    df = pd.read_parquet(f"../src/output/flight_{i}.parquet")
    flights.append(df)

for i in range(N):
    for j in range(N):
        print(i, " ", j)

        w_vals = []

        for col in columns:
            x = flights[i][col].values
            y = flights[j][col].values

            x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

            dist = wasserstein_distance(x_norm, y_norm)
            w_vals.append(dist)

        matrix[i, j] = np.mean(w_vals)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("Wasserstein distance")
plt.show()
