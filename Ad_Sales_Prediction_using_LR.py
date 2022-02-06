#!/usr/bin/env python
# coding: utf-8

# # Importing Lib

# In[1]:


import pandas as pd
import numpy as np


# # Loading Dataset

# In[9]:


data1 = pd.read_csv("DigitalAd_dataset.csv")
print(data1.shape)
print(data1.head(5))


# # Segregrate Dataset into X and Y

# In[10]:


x = data1.iloc[:,:-1].values
x


# In[11]:


y = data1.iloc[:,-1].values
y


# # Splitting Dataset into Train n Test

# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)


# # Feature Scaling

# In[15]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # Training using LR

# In[17]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train,y_train)


# # Prediction by adding new Customer

# In[18]:


age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))
newCust = [[age,sal]]
result = model.predict(sc.transform(newCust))
print(result)
if result == 1:
    print("Customer will buy")
else:
    print("Customer won't buy")


# # Prediction of all test data

# In[20]:


y_predict = model.predict(x_test)
print(np.concatenate((y_predict.reshape(len(y_predict),1),y_test.reshape(len(y_test),1)),1))


# # Finding Accurracy of the model

# In[21]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_predict)
print(cm) #printing confusion matrix
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test,y_predict)*100))


# # End of Module
