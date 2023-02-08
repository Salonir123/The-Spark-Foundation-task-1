#!/usr/bin/env python
# coding: utf-8
# # GRIP: The Spark Foundation
#
# ## Data Science and Buisness Analytic Internship
# 
# ## Task 1: Prediction Using Supervised ML
# 
# ### In this task we have to predict the percentage score of a student baased on the no. of hours studied. The task has two variables where ythe feature is the no. of hours studied and the target value is the percentagr score. this can be solved using simple linear regression.
# 
# ## Author: Patil Saloni Ravindra
# ## Step-1 import required libraries
# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# In[2]:
data=pd.read_csv("C:/Users/Abhijeet Gurav/Desktop/task 1 dataset.csv")
data
# ## Step -2 Exploring the data
# In[3]:
data.shape
# In[4]:
data.describe()
# In[5]:
data.info()
# ## Step-3 Data Visualization
# In[6]:
data.plot(kind='scatter',x='hours',y='scores');
plt.show()
# In[7]:
data.corr(method='pearson')
# In[8]:
data.corr(method='spearman')
# In[9]:
hours=data['hours']
scores=data['scores']
# In[10]:
sns.distplot(hours)
# In[11]:
sns.distplot(scores)
# ## Step-4 Linear Regression
# In[12]:
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
# In[13]:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)
# In[14]:
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train, y_train)
# In[15]:
m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line);
plt.show()
# In[16]:
y_pred=reg.predict(x_test)
# In[17]:
actual_predicted=pd.DataFrame({'Target':y_test,'predicted':y_pred})
actual_predicted
# In[18]:
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()
# what would be the predicted score if student studies for 9.25 hours/day?
# In[19]:
h=9.25
s=reg.predict([[h]])
print("If student studies for {} hours per day he/she will score {}% in exam.".format(h,s))
# ## Step-5 Model Evolution
# In[20]:
from sklearn import metrics
from sklearn.metrics import r2_score
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score:',r2_score(y_test,y_pred))
# ## Thank you!
