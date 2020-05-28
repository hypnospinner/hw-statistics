import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sc

df = pd.DataFrame({
    '0':pd.Series(['3','12']), 
    '2':pd.Series([6, 24]),
    '3':pd.Series([1, 4])
})

df.index = ['1st', '2nd']
print(df.head())
# OUTPUT:
#       0   2  3
# 1st   3   6  1
# 2nd  12  24  4