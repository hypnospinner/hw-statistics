import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sc

data = []

for i in range(3):
    data.append(np.mean([0, 10]))

for i in range(8):
    data.append(np.mean([10, 15]))
    
for i in range(10):
    data.append(np.mean([15, 20]))
    
for i in range(18):
    data.append(np.mean([20, 25]))
    
for i in range(21):
    data.append(np.mean([25, 30]))
    
for i in range(20):
    data.append(np.mean([30, 35]))

for i in range(11):
    data.append(np.mean([35, 40]))

for i in range(9):
    data.append(np.mean([40, 45]))

series = pd.Series(data)
df = pd.DataFrame({
    'Mean':pd.Series(series.mean()),
    'Std':pd.Series(series.std()),
    '0.25': pd.Series(series.quantile(0.25)),
    'Median':pd.Series(series.median()),
    '0.75': pd.Series(series.quantile(0.75)),
    'IQR':pd.Series(series.quantile(0.75)-series.quantile(0.25))
})

print(df.head())
# OUTPUT:
#      Mean       Std  0.25  Median  0.75   IQR
# 0  27.175  9.202101  22.5    27.5  32.5  10.0