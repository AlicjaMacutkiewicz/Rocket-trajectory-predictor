# similarity of data but include time shifts

# todo znalezc komputer ktory to uciagnie (BARDZO duzo pamieci)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


def dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])

            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]


N = 12
matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        df1 = pd.read_parquet(f"../src/output/flight_{i}.parquet")
        df2 = pd.read_parquet(f"../src/output/flight_{j}.parquet")

        print(i, " ", j)

        distances = []

        for col in columns:
            x = df1[col].values
            y = df2[col].values

            x = (x - x.mean()) / (x.std() + 1e-8)
            y = (y - y.mean()) / (y.std() + 1e-8)

            d = dtw_distance(x, y)
            distances.append(d)

        score = np.mean(distances)
        print(score)

        matrix[i, j] = distances

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=[f"F{i}" for i in range(N)],
    yticklabels=[f"F{i}" for i in range(N)],
)

plt.title("Podobieństwo lotów (Pearson/Spearman correlation)")
plt.show()
