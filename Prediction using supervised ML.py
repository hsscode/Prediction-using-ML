#!/usr/bin/env python
# coding: utf-8

# **IMPORT ALL LIBERARIES WHICH IS REQUIRED FOR THIS TASK**

# In[ ]:


pip install seaborn
get_ipython().system('pip install scikit-learn')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit-learn as sklearn


# **Read the csv data**

# In[8]:


data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[13]:


data.head()


# In[15]:


data.shape


# In[17]:


data.info()


# In[18]:


data.describe()


# **Visualise the data**

# In[24]:


sns.scatterplot(x=data['Hours'], y= data['Scores']);


# In[25]:


sns.regplot(x=data['Hours'], y= data['Scores']); #Regression plot for better understanding 


# **Seprate feature and targets**

# In[58]:


X=data[['Hours']]
y=data['Scores']


# **Train test split**

# In[66]:


from sklearn.model_selection import train_test_split
train_X, value_X, train_y, value_y = train_test_split(X,y, random_state=0)


# In[40]:


#model building 


# In[67]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()


# In[90]:


regressor.fit(train_X.values, train_y.values) #training the model


# In[95]:


pred_y=regressor.predict(value_X)


# In[92]:


pd.DataFrame({'Actual':value_y,'Predicted':pred_y})


# **Actual vs predicted Distribution**

# In[83]:


sns.kdeplot(pred_y,label="Predicted", shade=True);
sns.kdeplot(data=value_y,label="Actual", shade=True);


# In[84]:


print('Train Accuracy: ', regressor.score(train_X, train_y),'\nTest Accuracy: ',regressor.score(value_X,value_y))


# **What will be the predicted score if a student studies for 9.25hrs/day?**

# In[100]:


h=[[9.25]]
s=regressor.predict(h)
print('A student who studies ', h[0][0], 'hours is estimated to score approx', s[0])

