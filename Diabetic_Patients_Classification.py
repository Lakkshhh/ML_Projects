#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('diabetes.csv')
df.head()


# In[3]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


x = df.iloc[:,:-1]     # df.drop('Outcome', axis=1)
y = df.iloc[:,-1]      # df['Outcome']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[9]:


x.head()


# In[10]:


y.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)    # 0.25 * 768 = 192 is size of test data
print(x_train.shape)                                                       # 0.75 * 768 = 576 is remaining
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[46]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[47]:


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix = \n',cm)
    print('Classification Report = \n',classification_report(ytest,ypred))
    print('Acc Score = ',accuracy_score(ytest,ypred))


# ## Build Models

# ### 1) Log Reg

# In[48]:


from sklearn.linear_model import LogisticRegression


# In[49]:


m1 = LogisticRegression(max_iter=1000)
m1.fit(x_train,y_train)


# In[50]:


print('Training Score = ', m1.score(x_train,y_train))
print('Testing Score = ', m1.score(x_test,y_test))


# In[51]:


ypred_m1 = m1.predict(x_test)
print(ypred_m1)


# In[52]:


gen_metrics(y_test,ypred_m1)


# ### 2) KNN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier


# In[121]:


m2 = KNeighborsClassifier(n_neighbors=11)
m2.fit(x_train,y_train)


# In[122]:


print('Training Score = ', m2.score(x_train,y_train))
print('Testing Score = ', m2.score(x_test,y_test))


# In[123]:


ypred_m2 = m2.predict(x_test)
print(ypred_m2)


# In[124]:


gen_metrics(y_test,ypred_m2)


# ### 3) DT

# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[117]:


m3 = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
m3.fit(x_train,y_train)


# In[118]:


print('Training Score = ', m3.score(x_train,y_train))
print('Testing Score = ', m3.score(x_test,y_test))


# In[119]:


ypred_m3 = m3.predict(x_test)
print(ypred_m3)


# In[120]:


gen_metrics(y_test,ypred_m3)


# ### 4) Random Forest

# In[72]:


from sklearn.ensemble import RandomForestClassifier


# In[109]:


m4 = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=6,min_samples_split=12)
m4.fit(x_train,y_train)


# In[110]:


print('Training Score = ', m4.score(x_train,y_train))
print('Testing Score = ', m4.score(x_test,y_test))


# In[111]:


ypred_m4 = m4.predict(x_test)
print(ypred_m4)


# In[112]:


gen_metrics(y_test,ypred_m4)


# ### 5) SVM

# In[104]:


from sklearn.svm import SVC


# In[105]:


m5 = SVC(kernel='linear',C=1)
m5.fit(x_train,y_train)


# In[106]:


print('Training Score = ', m5.score(x_train,y_train))
print('Testing Score = ', m5.score(x_test,y_test))


# In[107]:


ypred_m5 = m5.predict(x_test)
print(ypred_m5)


# In[108]:


gen_metrics(y_test,ypred_m5)


# ## Conclusion
# #### 1) Log_Reg is the best performing model in terms of accuracy.<br>
# #### 2) Random Forest is the best performing model in terms of recall.<br>
