import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sc

data = {
    's_1': pd.Series(['19', '31', '34', '35', '39', '39', '43']),
    's_2': pd.Series(['7', '9', '15', '16', '16', '18', '22', '25', '27', '33', '39'])}
dataset = pd.DataFrame(data)
plot = sb.boxplot(data=dataset, palette='rainbow', orient='h')
plot.figure.savefig('task_3_plot.png')
