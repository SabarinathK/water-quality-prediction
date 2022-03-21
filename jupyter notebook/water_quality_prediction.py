#!/usr/bin/env python
# coding: utf-8

# # Water quality prediction

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC

import joblib as jb


import warnings
warnings.filterwarnings("ignore")


# ### Data Read

# In[2]:


df=pd.read_csv('water_potability.csv')


# In[3]:


df


#  1 means Potable(good for drinking) and 0 means Not potable.(not a drinking water)

# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.nunique()


# In[7]:


# Finding Null values
null=df.isnull().sum()
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull())


# ph , Sulfate,Trihalomethanes columns has null values
# 
# # Replacing null value with median

# In[8]:


df=df.fillna(df.median())


# In[9]:


df.isnull().sum()


# ### FInding Outliers

# In[10]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df)


# In[11]:


for column in df.columns[:-1]:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=(column),data=df)


# # handling oultier

# In[12]:


for column in df.columns[:-1]:
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_range=Q1-(IQR*1.5)
    upper_range=Q3+(IQR*1.5)
    df[column]=np.where(df[column]<lower_range,lower_range,df[column])
    df[column]=np.where(df[column]>upper_range,upper_range,df[column])


# In[13]:


for column in df.columns[:-1]:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=(column),data=df)


# In[ ]:





# In[14]:


for column in df.columns[:-1]:
    plt.figure(figsize=(10,5))
    sns.histplot(x=(column),data=df)


# ### dataset balanced or imbalanced

# In[15]:


df['Potability'].value_counts()


# dataset is imbalanced

# In[16]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)


# # data separation

# In[17]:


df.columns


# In[18]:


y=df['Potability'].copy()
X=df.drop('Potability',axis=1).copy()


# In[19]:


X.head()


# In[20]:


y.head()


# In[21]:


y.value_counts()


# In[22]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))


# In[23]:


y_res.value_counts()


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,train_size=0.7,random_state=356)


# In[ ]:





# In[25]:


X_train.shape,X_test.shape


# In[26]:


y_train.shape,y_test.shape


# In[27]:


y_res.value_counts()


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC


# In[29]:


from sklearn.metrics import classification_report, plot_confusion_matrix


# In[30]:


def algo(model):

    model=model
    model.fit(X_res,y_res)
    y_pred=model.predict(X_test)
    print('____________________________________\n',model,'Report\n____________________________________')
    print('train_accuracy',model.score(X_res,y_res)*100)
    print('test_accuracy',model.score(X_test,y_test)*100)
    print(classification_report(y_pred,y_test))
    print(plot_confusion_matrix(model,X_test,y_test))


# In[31]:


model=LogisticRegression()
algo(model)


# In[ ]:





# In[32]:


algo(DecisionTreeClassifier())


# In[33]:


algo(RandomForestClassifier())
jb.dump(model,'model_rfc.pkl')


# In[34]:


algo(AdaBoostClassifier())


# In[35]:


algo(GradientBoostingClassifier())


#  Random Forest is Good when compare to other model performance
#  so Random forest is saved
