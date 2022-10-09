# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:29:06 2022

@author: TwoYoung
"""

import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler


os.chdir(r"C:\Users\TwoYoung\OneDrive\OneDrive - University of North Carolina at Chapel Hill\Research\2022CDC\Data")
rawdf=pd.read_csv("sf_flood_cleaned.csv")

# K means clustering
X_Kmeans_df=rawdf[rawdf['FloodHealthIndex_Quintiles']>3]
X_Kmeans_df2=X_Kmeans_df.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles'],axis=1)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))
visualizer.fit(X_Kmeans_df2) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure
plt.show()

K = KMeans(n_clusters=3, random_state=0).fit(X_Kmeans_df2)
labels = K.labels_
X_Kmeans_df['K_means_labels']=labels
X_Kmeans_df.to_csv("sf_flood_kmeans.csv")

# Similarity inside each cluster (RSD method)
for i in range(0,3):
    print ("Relative Standard Deviations of Varibales in Cluster " + str(i+1))
    cluster_df=X_Kmeans_df[X_Kmeans_df["K_means_labels"]==i]
    cluster_df=cluster_df.drop(['GEOID',"K_means_labels","FloodHealthIndex","FloodHealthIndex_Quintiles"],axis=1)
    stdlist=(cluster_df.std()/cluster_df.mean()).sort_values()
    print(stdlist)
    
    
#randomforest
X_df=rawdf.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles'],axis=1)
y_df=rawdf['FloodHealthIndex']

X_train,X_test,y_train,y_test = train_test_split(X_df, y_df, random_state=1991,test_size=0.3)

feature_names = [f"feature {i}" for i in range(X_df.shape[1])]
forest = RandomForestRegressor(random_state=0)
forest.fit(X_train, y_train)

score = forest.score(X_train, y_train)
print("R-squared:", score) 

y_pred = forest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0)) 

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=X_df.columns)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.grid(False)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

#randomforest (exclude natural risks)
X_df=rawdf.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles','Elevation','SeaLevelRise','Precipitation'],axis=1)
y_df=rawdf['FloodHealthIndex']

X_train,X_test,y_train,y_test = train_test_split(X_df, y_df, random_state=1991,test_size=0.3)

feature_names = [f"feature {i}" for i in range(X_df.shape[1])]
forest = RandomForestRegressor(random_state=0)
forest.fit(X_train, y_train)

score = forest.score(X_train, y_train)
print("R-squared:", score) 

y_pred = forest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0)) 

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=X_df.columns)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI (Natural Risk Variables Excluded)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


#Home Price Correlation Test
corrdf=rawdf.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles'],axis=1)
corrlist=corrdf.corr()[['Elevation','SeaLevelRise','Precipitation']]
print(corrlist)


#linear regression
from statsmodels.api import OLS
X_df=rawdf.drop(['GEOID','FloodHealthIndex', 'FloodHealthIndex_Quintiles', 'Homeless'],axis=1)
y_df=rawdf['Homeless']
OLS(y_df,X_df).fit().summary()
y_df=rawdf['SeaLevelRise']
OLS(y_df,X_df).fit().summary()
y_df=rawdf['Precipitation']
OLS(y_df,X_df).fit().summary()


    
#Home Price Correlation Test
homepricedf=pd.read_csv("sf_flood_homeprice.csv")
homepricedf=homepricedf.drop("GEOID_1",axis=1)
corrlist=homepricedf.corr()['Homeprice'].sort_values(ascending=False)
print(corrlist)

