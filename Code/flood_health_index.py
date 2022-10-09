#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer as kv


# In[2]:


data = pd.read_csv("sf_flood_cleaned.csv")


# In[3]:


data.head()


# In[21]:


for i in list(data.columns)[1:]:
    if i not in ["Elevation","GEOID","SeaLevelRise","Precipitation","FloodHealthIndex","FloodHealthIndex_Quintiles"]:
        data.plot(kind='scatter',x="Elevation", y=i, c=color.reshape(1,-1))
        plt.show()


# In[22]:


for i in list(data.columns)[1:]:
    if i not in ["Elevation","GEOID","SeaLevelRise","Precipitation","FloodHealthIndex","FloodHealthIndex_Quintiles"]:
        data.plot(kind='scatter',x="SeaLevelRise", y=i, c=color.reshape(1,-1))
        plt.show()


# In[24]:


for i in list(data.columns)[1:]:
    if i not in ["Elevation","GEOID","SeaLevelRise","Precipitation","FloodHealthIndex","FloodHealthIndex_Quintiles"]:
        data.plot(kind='scatter',x="Precipitation", y=i, c=color.reshape(1,-1))
        plt.show()


# In[27]:


data.corr()


# In[35]:


kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(data)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

print(labels)


# In[53]:


data[(data['FloodHealthIndex_Quintiles']>3)]
k_data=data.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles'],axis=1)

K = KMeans(n_clusters=4, random_state=0).fit(k_data)
labels = K.labels_
print(labels)
k_data['K_means_labels']=labels
k_data['GEOID']=data['GEOID']
k_data.to_csv("k_means_output.csv")
print(k_data.head())
model = KMeans()
visualizer = kv(model, k=(1, 11))

visualizer.fit(k_data) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure
plt.show()


# In[ ]:




