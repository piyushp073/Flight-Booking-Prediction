#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv('Flight_Booking.csv')

# Remove the "Unnamed: 0" column
df = df.drop(columns='Unnamed: 0')

df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum


# In[7]:


import pandas as pd
import seaborn as sns
plt.figure(figsize=(10,8));
# Create a line plot in seaborn
sns.lineplot(data=df, x='airline', y='price')


# Add title and axis labels
sns.set_style("darkgrid")
plt.title('Airlines Vs Price')
plt.xlabel('Airline')
plt.ylabel('Price')

# Show the plot
plt.show()


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8));
# Create the line plot
sns.lineplot(data=df, x='days_left', y='price')

# Add a title and axis labels
plt.title('Days Left For Departure Versus Ticket Price')
plt.xlabel('Days Left For Departure')
plt.ylabel('Ticket Price')

# Display the plot
plt.show()


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
plt.figure(figsize=(15,8));
# Create the bar plot
sns.barplot(data=df, x='airline', y='price')

# Add a title and axis labels
plt.title('Airline Versus Ticket Price')
plt.xlabel('Airline')
plt.ylabel('Ticket Price')

# Display the plot
plt.show()



# In[10]:


plt.figure(figsize=(15,8));
sns.barplot(x='class',y='price' , data=df,hue='airline')


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
df = pd.read_csv('Flight_Booking.csv')

# Remove the "Unnamed: 0" column
df = df.drop(columns='Unnamed: 0')

# Create the subplots
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.lineplot(x='days_left', y='price', data=df, hue='source_city', ax=ax[0])
sns.lineplot(x='days_left', y='price', data=df, hue='destination_city', ax=ax[1])

# Add a title and axis labels
ax[0].set_title('Ticket Price by Days Left for Departure (Source City)')
ax[0].set_xlabel('Days Left for Departure')
ax[0].set_ylabel('Ticket Price')
ax[1].set_title('Ticket Price by Days Left for Departure (Destination City)')
ax[1].set_xlabel('Days Left for Departure')
ax[1].set_ylabel('Ticket Price')

# Display the plot
plt.show()


# In[12]:


plt.figure(figsize=(15,23));

plt.subplot(4,2,1)
sns.countplot(x=df["airline"],data=df)
plt.title("Frequency of Airline")


# In[13]:


df


# In[14]:


plt.figure(figsize=(15,23));

plt.subplot(4,2,1)
sns.countplot(x=df["airline"],data=df)
plt.title("Frequency of Airline")


plt.show()


# In[15]:


plt.figure(figsize=(15,23));

plt.subplot(4,2,1)
sns.countplot(x=df["source_city"],data=df)
plt.title("Frequency of source_city")


# In[16]:


plt.figure(figsize=(15,23));

plt.subplot(4,2,1)
sns.countplot(x=df["stops"],data=df)
plt.title("Frequency of stops")


# In[17]:


plt.figure(figsize=(15,23));
plt.subplot(4,2,1)
sns.countplot(x=df["arrival_time"],data=df)
plt.title("Frequency of arrival_time")


# In[18]:


plt.figure(figsize=(15,23));
plt.subplot(4,2,1)
sns.countplot(x=df["destination_city"],data=df)
plt.title("Frequency of destination_city")


# In[19]:


plt.figure(figsize=(15,23));
plt.subplot(4,2,1)
sns.countplot(x=df["class"],data=df)
plt.title("Frequency of class")


# In[21]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["airline"] = le.fit_transform(df["airline"])

df["source_city"] = le.fit_transform(df["source_city"])

df["departure_time"] = le.fit_transform(df["departure_time"])

df["stops"] = le.fit_transform(df["stops"])

df["arrival_time"] = le.fit_transform(df["arrival_time"])

df["destination_city"] = le.fit_transform(df["destination_city"])

df["class"] = le.fit_transform(df["class"])

df.info()


# In[22]:


plt.figure(figsize=(18,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'price')):
        col_list.append(col)

X = df[col_list]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

vif_data = vif_data[vif_data['VIF'] < 5]
X = X.drop('stops', axis=1)

print(vif_data)


# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

x = df.drop(columns=["price"])
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

difference = pd.DataFrame(np.c_[y_test, y_pred], columns=["Actual_Value", "Predicted_value"])
difference.head()


# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

x = df.drop(columns=["price"])
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

difference = pd.DataFrame(np.c_[y_test, y_pred], columns=["Actual_Value", "Predicted_value"])


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(y_test, label="Actual")
sns.distplot(y_pred, label="Predicted")
plt.legend()
plt.show()


# In[31]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

r2 = r2_score(y_test, y_pred)

mean_abs_error = mean_absolute_error(y_test, y_pred)
mean_abs_perc_error = mean_absolute_percentage_error(y_test, y_pred)

mean_sq_error = mean_squared_error(y_test, y_pred)
root_mean_sq_error = np.sqrt(mean_sq_error)

print("Mean Absolute Percentage Error:", mean_abs_perc_error)
print("Root Mean Squared Error:", root_mean_sq_error)


# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mean_abs_error = metrics.mean_absolute_error(y_test, y_pred)
mean_sq_error = metrics.mean_squared_error(y_test, y_pred)
root_mean_sq_error = np.sqrt(mean_sq_error)

mean_abs_perc_error = metrics.mean_absolute_percentage_error(y_test, y_pred) * 100

print("Random Forest Regressor:\n")
print("R-squared:", r2)
print("Mean Absolute Error:", mean_abs_error)
print("Mean Squared Error:", mean_sq_error)
print("Root Mean Squared Error:", root_mean_sq_error)
print("Mean Absolute Percentage Error:", mean_abs_perc_error, "%")

sns.distplot(y_test, label="Actual")
sns.distplot(y_pred, label="Predicted")
plt.legend()
plt.show()


# In[ ]:




