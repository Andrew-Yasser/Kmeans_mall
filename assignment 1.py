#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt


data_set = pd.read_csv("Mall_Customers.csv")
data_set.drop_duplicates(inplace= True)
x = data_set.iloc[:, [3,4]].values
wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
    k_means.fit(x)
    wcss.append(k_means.inertia_)
    


k_means = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = k_means.fit_predict(x)
for i in range(5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], label = "cluster" + str(i+1))
    plt.legend()
plt.grid(False)

plt.show()



# In[ ]:




