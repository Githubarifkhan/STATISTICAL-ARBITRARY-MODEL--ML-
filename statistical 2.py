#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('classic')
##%matplotlib inline


# In[16]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30,20

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import TheilSenRegressor, LinearRegression, RANSACRegressor


from sklearn.metrics import recall_score, precision_score


# In[17]:


stocks = pd.read_csv(r"C:\Users\arif0786\20microns.csv")
print(stocks.head())


# In[18]:


stocks.tail()


# In[19]:


stocks


# In[20]:


#Time Series Analysis
start16 = datetime(2016, 1, 1)
end16 = datetime(2016, 12, 31)
stamp16 = pd.date_range(start16, end16)

start17 = datetime(2017, 1, 1)
end17 = datetime(2017, 12, 31)
stamp17 = pd.date_range(start17, end17)

stocks['Date'] = pd.to_datetime(stocks.TIMESTAMP,format='%Y-%m-%d')
stocks.index = stocks['Date']


# In[21]:


#New Dataset
stocks = stocks[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'Date', 'PREVCLOSE', 'TOTTRDVAL', 'TOTALTRADES']]
stocks['HL_PCT'] = (stocks['HIGH'] - stocks['LOW']) / stocks['LOW'] * 100.0
stocks.index = stocks['Date']


# In[22]:


#Seperating Train and test data
train = []
test = []
for index, rows in stocks.iterrows():
    if index in stamp16:
        train.append(list(rows))
    if index in stamp17:
        test.append(list(rows))

train = pd.DataFrame(train, columns = stocks.columns)
test = pd.DataFrame(test, columns = stocks.columns)


# In[23]:


print(train.head())


# In[24]:


print(test.head())


# In[25]:


#Pre-Processing  Train Data 
X_train = train[['HIGH', 'LOW', 'OPEN', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']]
x_train = X_train.to_dict(orient='records')
vec= DictVectorizer()
X = vec.fit_transform(x_train).toarray()
Y = np.asarray(train.CLOSE)
Y = Y.astype('int')


# In[26]:


#Pre-Processing Test data
X_test = test[['HIGH', 'LOW', 'OPEN', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']]
x_test = X_test.to_dict(orient='records')
vec = DictVectorizer()
x = vec.fit_transform(x_test).toarray()
y = np.asarray(test.CLOSE)
y = y.astype('int')


# In[ ]:





# In[27]:


#Classifier
clf = TheilSenRegressor()
clf.fit(X, Y) 
print("Accuracy of this Statistical Arbitrage model is: ",clf.score(x,y))
predict = clf.predict(x)

test['predict'] = predict


# In[28]:


#Ploting 
train.index = train.Date
test.index = test.Date
train['CLOSE'].plot()
test['CLOSE'].plot()
test['predict'].plot()
plt.legend(loc='best')
plt.title('STATISTICAL ARBITRAGE MODEL PREDICTION OF NSE DATA 2016 AND 2017')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




