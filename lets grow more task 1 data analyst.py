#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


df=pd.read_csv("iris.csv")


# In[37]:


df.head() #top 5 values


# In[38]:


df.tail() #last 5 values show


# In[39]:


df.shape #no of rows and coloums


# In[40]:


df.isnull()


# In[41]:


df.isnull().sum()


# In[42]:


df.describe()


# In[43]:


df.columns


# In[44]:


df.nunique()


# In[45]:


df.variety.nunique()


# In[46]:


df.variety.value_counts()


# In[47]:


df.max()


# In[48]:


df.min()


# In[50]:


#the boxplot plot is related with the boxplot() method. The example below loads the iris flower data set.(a box-whisker plot)
#Then the presented boxplot shows the minimum, maximum, mediun. 1st quatite and 3ed quartite
sns.boxplot(x="variety",y="petal.length", data=df)
plt.show()


# In[51]:


sns.boxplot(x="variety",y="sepal.width", data=df)
plt.show()


# In[52]:


sns.boxplot(x="variety",y="sepal.length",data=df)
plt.show()


# In[53]:


sns.boxplot(x="variety",y="petal.width", data=df)
plt.show()


# In[54]:


sns.boxplot(y="sepal.length",data=df)
plt.show()


# In[55]:


sns.boxplot(y="sepal.width",data=df)
plt.show()


# In[56]:


sns.boxplot(y="petal.length",data=df)
plt.show()


# In[57]:


sns.boxplot(y="petal.width",data=df)
plt.show()


# In[58]:


sns.pairplot(df,hue="variety") #a pairplot plot a pairwise relationship in a dataset


# In[59]:


#heatup using to show 20data in graphical format.Each data value represent in a matrix and it has a special color
#'True' value to annat the the value will show on each cell of the heatmap
#we change the color of seaborn heatmap but center parameter will change cmap according to give value by the creator
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap="seismic")
plt.show()


# In[63]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() #LableEncoder can be use normalize lable


# In[64]:


df['variety']=le.fit_transform(df['variety'])
df.head()


# In[65]:


x=df.drop(columns=['variety']) #drop column
y=df['variety']
x[:5]


# In[66]:


y[:5]


# In[67]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[69]:


lr=LogisticRegression()
knn=KNeighborsClassifier
snm=SVC()
nb=GaussianNB()
dt=DecisionTreeClassifier
rf=RandomForestClassifier


# In[70]:


models=[lr,knn,snm,nb,dt,rf]
scores=[]


# In[77]:


for model in models:
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred)
    print("Accuracy af" + type(model).__name__+ "is",accuraccy_score(y_test,y_pred))
                  


# In[ ]:





# In[ ]:




