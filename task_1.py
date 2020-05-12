import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sc

data = []

for i in range(2):
    data.append(99.5)

for i in range(16):
    data.append(100)

for i in range(22):
    data.append(100.5)

for i in range(10):
    data.append(101)

# convert data to series
data_s = pd.Series(data)

# build dataframe
df = pd.DataFrame(data_s, columns=['weight (g)'])
print(df.describe())
# OUTPUT:
#        weight (g)
# count   50.000000
# mean   100.400000
# std      0.404061
# min     99.500000
# 25%    100.000000
# 50%    100.500000
# 75%    100.500000
# max    101.000000

# in grams
variation = df.std() / df.mean()
print(variation)
# OUTPUT:
# weight (g)    0.004025
# dtype: float64

kg_df = pd.DataFrame(data_s * 0.001, columns=['weight (kg)'])

print(kg_df.describe())
# OUTPUT:
#        weight (kg)
# count    50.000000
# mean      0.100400
# std       0.000404
# min       0.099500
# 25%       0.100000
# 50%       0.100500
# 75%       0.100500
# max       0.101000

# kg variation
kg_variation = kg_df.std() / kg_df.mean()
print(kg_variation)
# OUTPUT:
# weight (kg)    0.004025
# dtype: float64

# distribution graph

ox = np.sort(df['weight (g)'])
oy = np.arange(len(ox)) / float(len(ox))

plt.plot(ox, oy)
# plt.show()

pdf = pd.DataFrame((data_s * 0.98) + 11, columns=['price'])
print(pdf.describe())
# OUTPUT:
#            price
# count   50.00000
# mean   109.39200
# std      0.39598
# min    108.51000
# 25%    109.00000
# 50%    109.49000
# 75%    109.49000
# max    109.98000

# task 2