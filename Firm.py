#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn import linear_model


# In[3]:


df=pd.read_csv(r"E:\AI\Data\Firm data\Dairy Firm Data-regression.csv")


# In[4]:


df.head()


# # Data Analysis

# In[5]:


from matplotlib import pyplot as plt 


# In[6]:


plt.xlabel('Ammount')
plt.ylabel('')
plt.title('Worker Cost In Firm')
plt.scatter(df.Worker,df.firm_name,color='black',linewidth=5, linestyle='dotted')


# In[7]:


plt.scatter(df.Current,df.firm_name,color='magenta')
plt.xlabel('Ammount')
plt.ylabel('')
plt.title('Electricity bills')


# In[8]:


plt.scatter(df.Water,df.firm_name,color='gray')
plt.xlabel('Ammount')
plt.ylabel('')
plt.title('Water Bills')


# In[ ]:





# In[ ]:





# In[9]:


df.head()


# In[ ]:





# In[10]:


df=pd.read_excel(r"E:\AI\Data\Firm data\All firms.xlsx")


# In[11]:


df.head(2)


# In[12]:


plt.bar(df.firm_name,df.Cow_to,color='green',)
plt.xlabel('Firm Name')
plt.ylabel('')
plt.title('Number of Cows')


# In[13]:


plt.bar(df.firm_name,df.Total_Fee_kg,color='red',)
plt.xlabel('Firm Name')
plt.ylabel('')
plt.title('Total Feed')


# In[14]:


import numpy as np
plt.scatter(df.firm_name,df.Total_Revenue, color=['red', 'green', 'green', 'green', 'green','green'],  s=70)
plt.xlabel('Firm Name')
plt.ylabel('')
plt.title('Total Revenue')


# In[ ]:





# In[ ]:





# # Prediction 

# In[15]:


df=df.drop(['firm_name','Cow_to'],axis=1)
df=df.dropna()


# In[16]:


df.head()


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


reg=LinearRegression()


# In[19]:


reg.fit(df[['Weight(kg)','Increase Weight','Total_Fee_kg','Corn(ভুট্টা)(kg)','Khail(খৈল)(kg)','Khor(খর)(kg)','Green Grass(kg)','Sailase(সাইলেস)(kg)','Wheat Husk(kg)','Madicine','Water','Current','Coil','Worker','Total_cost(tk)']],df[['Milk(Ltr.)','kids income','kids cost','Milk Income','Milk Revenue','Kids Revenue','Total_Revenue']])


# In[20]:


reg.predict([['470','471','20.59','2.2','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[21]:


df.head(2)


# # Quick Prediction

# In[22]:


df.head()


# In[23]:


df=df.drop(['Corn(ভুট্টা)(kg)','Khail(খৈল)(kg)','Khor(খর)(kg)','Green Grass(kg)','Sailase(সাইলেস)(kg)','Wheat Husk(kg)'],axis=1)
df=df.dropna()


# In[24]:


df.head()


# In[25]:


df=df.drop(['Madicine','Water'],axis=1)
df=df.dropna()


# In[26]:


df.head()


# In[27]:


reg.fit(df[['Weight(kg)','Increase Weight','Total_Fee_kg','Current','Coil','Worker','Total_cost(tk)']],df[['Milk(Ltr.)','kids income','kids cost','Milk Income','Milk Revenue','Kids Revenue','Total_Revenue']])


# In[28]:


reg.predict([[360,361,12,3.33,10.0,66,474]])


# In[ ]:





# # Loss and profit prediction

# In[29]:


import pandas as pd


# In[30]:


df=pd.read_csv(r"E:\AI\Data\Firm data\Dairy Firm Data-classification.csv")


# In[31]:


y =df[['Milk(Ltr.)','kids income','kids cost','Milk Income','Milk Revenue','Kids Revenue','Total_Revenue']]
x = df.drop(['profit','Cow no','Milk(Ltr.)','kids income','kids cost','Milk Income','Milk Revenue','Kids Revenue','Total_Revenue'],axis=1)
x=x.dropna()


# In[32]:


x


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


df.head()


# In[ ]:





# In[37]:


df.head()


# # Linear Regression

# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


logmodel=LinearRegression()


# In[40]:


logmodel.fit(x_train, y_train)


# In[41]:


logmodel.predict([['380','382.12','12.59','2.2','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[42]:


logmodel.score(x_test,y_test)


# In[43]:


logmodel.coef_


# In[44]:


logmodel.intercept_


# In[45]:


logmodel.score(x_train,y_train)


# # k-nearest neighbors algorithm

# In[46]:


from sklearn.neighbors import KNeighborsRegressor


# In[47]:


log = KNeighborsRegressor(n_neighbors=2)


# In[48]:


log.fit(x_train, y_train)


# In[143]:


log.predict([['353','353.12','10.59','2.1','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[50]:


log.score(x_test,y_test)


# # Lasso Regression

# In[51]:


from sklearn import linear_model


# In[52]:


clf = linear_model.Lasso(alpha=0.1)


# In[53]:


clf.fit(x_train, y_train)


# In[54]:


clf.predict([['368','373.12','9.59','4.2','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[55]:


clf.score(x_test,y_test)


# In[56]:


print(clf.coef_)


# In[57]:


print(clf.intercept_)


# # Ridge Regression

# In[58]:


from sklearn.linear_model import Ridge


# In[59]:


n_samples, n_features = 10, 5


# In[60]:


clf = Ridge(alpha=1.0)


# In[61]:


clf.fit(x_test,y_test)


# In[62]:


clf.predict([['353','353.12','10.59','2.2','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[63]:


clf.score(x_test,y_test)


# In[64]:


print(clf.coef_)


# In[65]:


print(clf.intercept_)


# In[ ]:





# # Gaussian Process Regressor

# In[66]:


from sklearn.gaussian_process import GaussianProcessRegressor


# In[67]:


from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


# In[68]:


kernel = DotProduct() + WhiteKernel()


# In[69]:


gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_test, y_test)


# In[70]:


gpr.score(x_test, y_test)


# In[71]:


gpr.predict([['353','353.12','10.59','2.2','1.29','0.89','3.18','1.90','2.11','30','6.66','3.33','10','66','360.55']])


# In[ ]:





# In[72]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = [ 'Gaussian', 'Lasso','Linear Regression','Ridge','KNN']
students = [90,96,97,97,98]
ax.bar(langs,students)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Of Algorithm')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[73]:


import pandas as pd


# In[74]:


df=pd.read_csv(r"E:\AI\Data\Firm data\Dairy Firm Data-classification.csv")


# In[75]:


df.head(2)


# In[76]:


y =df[['profit']]
x = df.drop(['profit','Cow no'],axis=1)
x=x.dropna()


# In[77]:


x


# In[78]:


y


# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[81]:


x_train


# In[82]:


y_train


# In[83]:


x_test


# In[84]:


y_test


# In[88]:


x_train.shape


# In[89]:


y_train.shape


# In[90]:


x_test.shape


# # Logistic Regression

# In[91]:


from sklearn.linear_model import LogisticRegression


# In[92]:


logmodel=LogisticRegression()


# In[93]:


logmodel.fit(x_train,y_train)


# In[94]:


predictions = logmodel.predict(x_test)


# In[95]:


from sklearn.metrics import classification_report


# In[96]:


classification_report(y_test,predictions)


# In[97]:


from sklearn.metrics import confusion_matrix


# In[98]:


confusion_matrix(y_test,predictions)


# In[99]:


from sklearn.metrics import accuracy_score


# In[100]:


accuracy_score(y_test,predictions)


# In[101]:


logmodel.score(x_test,y_test)


# # Naive Bayes

# In[102]:


from sklearn.naive_bayes import GaussianNB


# In[103]:


logmodel =GaussianNB()


# In[104]:


logmodel.fit(x_train, y_train)


# In[105]:


predictions = logmodel.predict(x_test)


# In[106]:


from sklearn.metrics import classification_report


# In[107]:


classification_report(y_test,predictions)


# In[108]:


from sklearn.metrics import confusion_matrix


# In[109]:


confusion_matrix(y_test,predictions)


# In[110]:


from sklearn.metrics import accuracy_score


# In[111]:


accuracy_score(y_test,predictions)


# # Decision Tree

# In[112]:


from sklearn import tree


# In[113]:


logmodel = tree.DecisionTreeClassifier()


# In[114]:


logmodel.fit(x_train, y_train)


# In[115]:


predictions = logmodel.predict(x_test)


# In[116]:


from sklearn.metrics import accuracy_score


# In[117]:


accuracy_score(y_test,predictions)


# In[118]:


from sklearn.metrics import classification_report


# In[119]:


classification_report(y_test,predictions)


# In[120]:


from sklearn.metrics import confusion_matrix


# In[121]:


confusion_matrix(y_test,predictions)


# # Support Vector Machines

# In[122]:


from sklearn import svm


# In[123]:


clf = svm.SVC()


# In[124]:


clf.fit(x_train, y_train)


# In[125]:


predictions = clf.predict(x_test)


# In[126]:


from sklearn.metrics import classification_report


# In[127]:


classification_report(y_test,predictions)


# In[128]:


from sklearn.metrics import confusion_matrix


# In[129]:


confusion_matrix(y_test,predictions)


# In[130]:


from sklearn.metrics import accuracy_score


# In[131]:


accuracy_score(y_test,predictions)


# # Random Forest Classifier

# In[132]:


from sklearn.ensemble import RandomForestClassifier


# In[133]:


rfc = RandomForestClassifier()


# In[134]:


rfc.fit(x_train, y_train)


# In[135]:


predictions = rfc.predict(x_test)


# In[136]:


from sklearn.metrics import classification_report


# In[137]:


classification_report(y_test,predictions)


# In[138]:


from sklearn.metrics import confusion_matrix


# In[139]:


confusion_matrix(y_test,predictions)


# In[140]:


from sklearn.metrics import accuracy_score


# In[141]:


accuracy_score(y_test,predictions)


# In[142]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Naive Bayes','Logistic Regression', 'Decision Tree','Support Vector Machines','Random Forest Classifier']
students = [47,100,100,100,100]
ax.bar(langs,students)
plt.xticks(fontsize=10, rotation='vertical')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Of Algorithm')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




