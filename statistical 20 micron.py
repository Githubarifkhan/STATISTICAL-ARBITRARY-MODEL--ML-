#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv(r"C:\Users\arif0786\nse_data.csv")

data = []

for index, rows in df.iterrows():
    if rows[0] == '20MICRONS':
        data.append(list(rows))

stocks = pd.DataFrame(data, columns = df.columns)

stocks.to_csv('20microns.csv', sep=',')

