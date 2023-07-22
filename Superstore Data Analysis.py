#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import os
print("libraries are imported successfully")


# In[4]:


# Reading Dataset
df_orders = pd.read_csv(r"C:\Users\Vishal Choudhary\Downloads\SampleSuperstore.csv")
print("Data Read Sucessfully")


# In[5]:


# Dimensions of Data 
df_orders.shape


# In[6]:


# Peek Of dataset
df_orders.head()


# In[7]:


#last 5 rows of dataset
df_orders.tail()


# In[8]:


df_orders.columns


# In[9]:


df_orders["Category"].unique()


# In[10]:


df_orders["Sub-Category"].unique()


# In[11]:


df_orders["Segment"].unique()


# In[12]:


df_orders=df_orders.drop(columns=["Postal Code"],axis =1)


# In[13]:


df_orders.head()


# In[14]:


df_orders["Category"].value_counts()


# In[15]:


df_orders["Category"].value_counts().sum()


# In[16]:


df_orders["Sub-Category"].nunique()


# In[17]:


df_orders["Sub-Category"].value_counts()


# In[18]:


df_orders["Region"].unique()


# In[19]:


df_orders["Ship Mode"].unique()


# In[20]:


df_orders["State"].unique()


# In[21]:


df_orders["City"].unique()


# In[22]:


df_orders.nunique()


# In[23]:


df_orders.info()


# In[24]:


df_orders.describe()


# In[25]:


df_orders.isnull().sum()


# In[26]:


print("total number of null values =",df_orders.isnull().sum().sum())


# In[27]:


df_orders.dtypes


# In[28]:


for col in df_orders:
    print(df_orders[col].unique())


# In[29]:


df_orders.duplicated().sum()


# In[30]:


sns.heatmap(df_orders.isnull(),yticklabels=False,cbar= False,cmap="viridis")


# In[31]:


df_orders.drop_duplicates()


# In[32]:


correlation= df_orders[['Sales','Quantity','Discount','Profit']].corr()
print(correlation)


# In[33]:


covarience= df_orders[['Sales','Quantity','Discount','Profit']].cov()
print(covarience)


# In[34]:


df_orders.iloc[0]


# In[35]:


df_orders.iloc[:,0]


# In[36]:


df_orders.value_counts()


# In[37]:


# Bar Plot Between Category and Sub Category
plt.figure(figsize = (16,8))
plt.bar('Sub-Category','Category',data=df_orders)
plt.show()


# In[38]:


# Pie Plot Of Sub Category
plt.figure(figsize=(12,10))
df_orders['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()
                                                  


# In[39]:


# Bar PLot Of Sales Vs Profit
df_orders.groupby('Sub-Category')[['Profit','Sales']].agg(['sum']).plot.bar()
plt.title('Total Profit and Sales Per Sub-Category')
plt.rcParams['figure.figsize']=[10,8]
plt.show()


# In[40]:


# Count Plot OF Cities
print(df_orders['State'].value_counts())
plt.figure(figsize=(15,8))
sns.countplot(x=df_orders['State'])
plt.xticks(rotation=90)
plt.show()


# In[41]:


# Sales and Profit Region Wise
df_orders.groupby('Region')[['Profit','Sales']].agg(['sum']).plot.bar()
plt.title('Total Profit and Sales Per Region')
plt.rcParams['figure.figsize']=[10,8]
plt.show()


# In[42]:


# Count Plot of Sub Category
print(df_orders['Sub-Category'].value_counts())
plt.figure(figsize=(15,8))
sns.countplot(x=df_orders['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[43]:


# Heat Map Of Correlation among The Columns
fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap( df_orders[['Sales','Quantity','Discount','Profit']].corr(),annot = True)
plt.show()


# In[44]:


#Heat Map Of Covarience among the Set of Variables
fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap( df_orders[['Sales','Quantity','Discount','Profit']].cov(),annot = True)
plt.show()


# In[45]:


#Count Plot Of Segment
sns.countplot(x=df_orders['Segment'])


# In[46]:


#Count Plot of Region
sns.countplot(x=df_orders['Region'])


# In[47]:


#Bar PLot of Sub- Category vs Profit
plt.figure(figsize =(40,25))
sns.barplot(x=df_orders['Sub-Category'],y=df_orders['Profit'])


# In[48]:


# Line Plot of Discount Vs Profit
plt.figure(figsize =(10,4))
sns.lineplot(x='Discount',y='Profit',data = df_orders,color ='r',label = 'Discount')
plt.legend()


# In[49]:


#Histogram of Data
df_orders.hist(bins=50,figsize=(20,15))
plt.show()


# In[50]:


# Pair- Plot of Sub_Category
figsize=(15,10)
sns.pairplot(df_orders,hue='Sub-Category')


# In[51]:


grouped= df_orders.groupby(['Ship Mode','Segment','Category','Sub-Category'])
grouped_sales_profit = grouped[['Sales','Profit']].sum().reset_index()
print(grouped_sales_profit)


# In[52]:


# Statistical Summary Of Data
df_orders.groupby("State").Profit.agg(["sum","mean","min","max","count","median","std","var"])


# In[53]:


#Pair Plot of Data
sns.pairplot(df_orders)


# In[54]:


# Box Plot of Sales
fig, axes = plt.subplots(figsize =(10,10))
sns.boxplot(df_orders['Sales'])


# In[55]:


# Box Plot Of Discount
fig, axes = plt.subplots(figsize =(10,10))
sns.boxplot(df_orders['Discount'])


# In[56]:


#Box Plot OF Profit
fig, axes = plt.subplots(figsize =(10,10))
sns.boxplot(df_orders['Profit'])


# In[57]:


df_orders.value_counts().nlargest().plot(kind = 'bar',figsize =(10,5))


# In[58]:


# Plot Of Value Count
fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(df_orders["Sales"], df_orders["Profit"])
ax.set_xlabel('Sales')
ax.set_ylabel('Profit')
plt.show()


# In[59]:


# Sales Statistical Data
print(df_orders['Sales'].describe())
plt.figure(figsize = (9,8))
plt.grid()
sns.distplot(df_orders['Sales'], color ='b',bins = 100 ,hist_kws = {'alpha': 0.4});


# In[60]:


#Cludstering of data
df_orders.plot(kind = 'box',subplots = True , layout =(3,2),sharex= False , sharey = False)
plt.rcParams['figure.figsize'] = [14,12]
plt.show()


# In[64]:


from sklearn.cluster import KMeans
x = df_orders[['Sales', 'Quantity']].values  

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(x)
    wcss.append(kmeans.inertia_)

sns.set_style("whitegrid")

sns.FacetGrid(df_orders, hue="Sub-Category", height=6).map(plt.scatter, "Sales", "Quantity")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='yellow', label='centroids')

plt.rcParams['figure.figsize'] = [10, 8]
plt.legend()
plt.show()


# In[ ]:




