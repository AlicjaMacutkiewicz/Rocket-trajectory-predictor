# compares each dataset 0 => not similar, 1 => equal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

columns = ['Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z', 'Best_AngVel_X',
       'Best_AngVel_Y', 'Best_AngVel_Z', 'Barometer_Value', 'Sensor_Value',
       'Thrust', 'Mass', 'Position_X', 'Position_Y', 'Position_Z',
       'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

N = 12
matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        print(i, " ", j)
        test1 = pd.read_parquet(f'../src/output/flight_{i}.parquet')
        test2 = pd.read_parquet(f'../src/output/flight_{j}.parquet')

        values = []

        for col in columns:
            _, p = stats.mannwhitneyu(test1[col], test2[col], alternative='two-sided')
            values.append(p)

        matrix[i, j] = np.median(values)

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=[f"F{i}" for i in range(N)],
    yticklabels=[f"F{i}" for i in range(N)]
)

plt.title("Flight similarity (Mann–Whitney p-value)")
plt.show()