import torch
import torchsummary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

df = pd.read_csv('https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-adjusted-index.csv')
df.date = pd.to_datetime(df.date)
df2 = df[(df['date'].dt.year==2019) & (df['date'].dt.month==1)]
#plt.figure(figsize=(5,4))
ax = sns.lmplot(x='GDP_dollar',y='dollar_price',data=df2,line_kws={'color': 'red'})
ax.set_xticklabels(rotation=45);