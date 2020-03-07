#!/usr/bin/env python
# coding: utf-8

# # Regression with Review Score Each Month

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[5]:


SingaporeAirBnb = pd.read_csv("listings.csv")
SingaporeAirBnb["reviews_per_month"] = SingaporeAirBnb["reviews_per_month"].replace(",", ".", regex=True)
SingaporeAirBnb = SingaporeAirBnb.dropna()
print(SingaporeAirBnb.dtypes)
print(SingaporeAirBnb.isna().values.any())


# In[9]:


newSingaporeAirBnb = SingaporeAirBnb[["neighbourhood","neighbourhood_group","room_type", "price", "minimum_nights", "number_of_reviews", 
                      "last_review", "reviews_per_month", "calculated_host_listings_count", "availability_365"]]

print(newSingaporeAirBnb.head())
newSingaporeAirBnb.hist()
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


# In[34]:


plt.subplot(331)
plt.scatter(newSingaporeAirBnb["availability_365"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("availability_365")
plt.ylabel("reviews_per_month")

plt.subplot(332)
plt.scatter(newSingaporeAirBnb["calculated_host_listings_count"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("calculated_host_listings_count")
plt.ylabel("reviews_per_month")

plt.subplot(333)
plt.scatter(newSingaporeAirBnb["neighbourhood"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("neighbourhood")
plt.ylabel("reviews_per_month")

plt.subplot(334)
plt.scatter(newSingaporeAirBnb["neighbourhood_group"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("neighbourhood_group")
plt.ylabel("reviews_per_month")

plt.subplot(335)
plt.scatter(newSingaporeAirBnb["room_type"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("room_type")
plt.ylabel("reviews_per_month")

plt.subplot(336)
plt.scatter(newSingaporeAirBnb["price"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("price")
plt.ylabel("reviews_per_month")

plt.subplot(337)
plt.scatter(newSingaporeAirBnb["minimum_nights"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("minimum_nights")
plt.ylabel("reviews_per_month")

plt.subplot(338)
plt.scatter(newSingaporeAirBnb["number_of_reviews"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("number_of_reviews")
plt.ylabel("reviews_per_month")

plt.subplot(339)
plt.scatter(newSingaporeAirBnb["last_review"], newSingaporeAirBnb["reviews_per_month"], color="blue")
plt.xlabel("last_review")
plt.ylabel("reviews_per_month")


# In[11]:


train, test = train_test_split(newSingaporeAirBnb, test_size=0.2)
regression = linear_model.LinearRegression()
regression.fit(train[["price"]], train[["reviews_per_month"]])
print('Coefficients: ', regression.coef_) 
print('Intercept: ',regression.intercept_)
print('Train: ',len(train))
print('Test: ',len(test))


# In[26]:


plt.scatter(train["price"], train["reviews_per_month"],  color='blue')
plt.plot(train[["price"]], regression.coef_ * train[["reviews_per_month"]] + regression.intercept_, '-r')
plt.ylabel("reviews_per_month")
plt.rcParams["figure.figsize"] = [10,8]
plt.show()


# In[28]:


sb.pairplot(train)
sb.lmplot("price", "reviews_per_month", data = train)
plt.show()


# In[29]:


prediction = regression.predict(test[["price"]])
for i in range(len(test)):
  print(test[["price"]].values[i], prediction[i])

print("MAE : ", mean_absolute_error(test[["reviews_per_month"]], prediction)) #Mean Absolute Error
print("MSE : ", mean_squared_error(test[["reviews_per_month"]], prediction)) #Mean Square Error
print("R2 : ", r2_score(test[["reviews_per_month"]], prediction)) #R2 Score


# # Classification - KNN

# In[30]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[31]:


SingaporeAirBnb = pd.read_csv("listings.csv")
SingaporeAirBnb["reviews_per_month"] = SingaporeAirBnb["reviews_per_month"].replace(",", ".", regex=True)
SingaporeAirBnb = SingaporeAirBnb.dropna()
# print(SingaporeAirBnb["neighbourhood"].unique())
SingaporeAirBnb["neighbourhood"] = pd.Categorical(SingaporeAirBnb["neighbourhood"], SingaporeAirBnb["neighbourhood"].unique())
SingaporeAirBnb["neighbourhood"] = SingaporeAirBnb["neighbourhood"].cat.rename_categories([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42])
print(SingaporeAirBnb.dtypes)
print(SingaporeAirBnb.isna().values.any())
print(SingaporeAirBnb.head())

train, test = train_test_split(SingaporeAirBnb, test_size=0.2)


# In[32]:


KNN = KNeighborsClassifier(n_neighbors = 3).fit(train[["price", "number_of_reviews", "minimum_nights", 
                                                       "reviews_per_month", "calculated_host_listings_count", 
                                                      "availability_365" ]], train["neighbourhood"])
classification = KNN.predict(test[["price", "number_of_reviews", "minimum_nights", 
                                                       "reviews_per_month", "calculated_host_listings_count", 
                                                      "availability_365"]])
accuracy = accuracy_score(test["neighbourhood"], classification)
MAE = mean_absolute_error(test["neighbourhood"], classification)
MSE = mean_squared_error(test["neighbourhood"], classification)

print(" ACC : %.2f" % accuracy)
print(" MAE : %.2f" % MAE)
print(" MSE : %.2f" % MSE)


# In[33]:


Ks = 10
accuracy = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1, Ks):    
    KNN = KNeighborsClassifier(n_neighbors = n).fit(train[["price", "number_of_reviews", "minimum_nights", 
                                                           "reviews_per_month", "calculated_host_listings_count", 
                                                           "availability_365"]], train["neighbourhood"])  
    classification = KNN.predict(test[["price", "number_of_reviews", "minimum_nights", "reviews_per_month", 
                                   "calculated_host_listings_count", "availability_365"]])
    accuracy[n - 1] = accuracy_score(test["neighbourhood"], classification)
    
print("Best  ACC : %.2f" % accuracy.max(), ", with k = ", accuracy.argmax() + 1)


# In[ ]:




