#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics

# ## Task-2 Prediction using Unsupervised machine Learning

# # By : IQRA MALIK

# ### Problem Statement: Predict the optimum number of clusters and represent it visually. 
# 

# #### Iris Dataset -  https://bit.ly/3kXTdox
#  

# In[174]:


#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


# In[175]:


#Loading the iris data alreeady present in sklearn library
iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns= iris.feature_names)
df.head()


# In[176]:


#for information of the dataset
#for describe the datsets to understand its relationships
df.info()
df.describe()


# In[177]:


#To check the no. of rows and columns of the dataset
df.shape


# In[178]:


#To check is there any null value of the dataset
df.isnull().sum()


# In[179]:


#Defining the correlation among all
df.corr()


# ## Data Visualization

# In[180]:


#Plotting Heatmap
fig= plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),linewidth=1,annot=True)


# In[181]:


#Plotting Histogram
df.hist(edgecolor="purple",linewidth=1.4)
fig=plt.gcf
plt.show()


# In[182]:


X = df.iloc[:,:] #independent features defining x


# In[183]:


#Standardization of the dataset
sc = StandardScaler()
X_std= sc.fit_transform(X)
X_std= pd.DataFrame(X_std)
X_std.columns = X.columns


# In[184]:


X_std


# # Deriving the K value using Elbow method

# In[185]:


#Importing K means from the Library
from sklearn.cluster import KMeans


# In[193]:


#Elbow Plotting
plt.figure(figsize=(10,6))
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X_std)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("Sum of squared error")
plt.grid()
plt.show()


# #### We can clearly see in the Above graph that the Elbow point is 3
# #### No. of clusters(K)=3

# ### Now training the Algorithm and fitting the Dataset

# In[187]:


#Declaring that K=3 and maximum iteration to be done is 100 and K-means++ for faster convergance
km= KMeans(n_clusters=3,max_iter=100,init="k-means++",random_state=5)
#Fitting model Prediction - with Standardization
y_kmeans = km.fit_predict(X_std)
y_kmeans


# In[188]:


#Fitting model - without Standardization
km2 = KMeans(n_clusters=3,max_iter=100,init="k-means++")
y_kmeans2 = km2.fit_predict(X)
y_kmeans2


# # Virtualization of the DataSet Prediction

# In[189]:


plt.figure(figsize=(12,6))

plt.scatter(X.iloc[y_kmeans2==1,0],X.iloc[y_kmeans2==1,1],s=50,c='red',label="Cluster1")
plt.scatter(X.iloc[y_kmeans2==2,0],X.iloc[y_kmeans2==2,1],s=50,c='purple',label="Cluster2")
plt.scatter(X.iloc[y_kmeans2==0,0],X.iloc[y_kmeans2==0,1],s=50,c='pink',label="Cluster3")
plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],marker="*",s=400,c='black',label="Centroids")


plt.title("Clusters of Species (Without Standardizing features)")
plt.xlabel("Features")
plt.ylabel("Clusters")
plt.legend()
plt.show


# ## Thank you
