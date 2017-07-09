
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ### Import training data

# In[2]:

train_df = pd.read_csv("data/training.txt", sep = '\t', encoding = 'iso-8859-1')


# In[48]:

pd.options.display.max_columns = 145


# In[49]:

train_df.head()


# In[50]:

train_df.ix[:,1:].describe()


# ### Preprocesssing 

# In[51]:

train_df.dtypes


# In[52]:

train_df['Year'] = pd.to_numeric(train_df['Year'].str.replace('Y',''), errors = 'coerce')


# In[53]:

train_df['GroundFloorArea'] = pd.to_numeric(train_df['GroundFloorArea'].str.replace('sq. m',''), errors = 'coerce')


# In[54]:

train_df.head()


# In[55]:

train_df.groupby('EnergyRatingCat')['EnergyRatingCat'].count().plot(kind = 'bar')


# In[56]:

train_df['Year'].corr(train_df['EnergyRatingCont'])


# In[57]:

train_df['GroundFloorArea'].corr(train_df['EnergyRatingCont'])


# In[58]:

train_df['Year'].corr(train_df['EnergyRatingCont'])


# In[59]:

numerics = ['float64', 'int64']
num_train_df = train_df.select_dtypes(include = numerics)


# In[60]:

ER = num_train_df['EnergyRatingCont']


# In[61]:

num_train_df.isnull().sum()


# In[62]:

cleaned_train_df = num_train_df.dropna(axis = 1, thresh = 100000)


# In[63]:

cleaned_train_df.fillna(cleaned_train_df.mean(), inplace = True)


# In[64]:

corr_list = cleaned_train_df.corrwith(ER).abs().sort_values(ascending = False) 


# In[65]:

rel_features = corr_list[corr_list > 0.2].index.tolist()
rel_features.append('BuildingID')
rel_features.append('EnergyRatingCat')


# In[66]:

cleaned_train_df1 = cleaned_train_df


# In[67]:

cleaned_train_df1['EnergyRatingCat'] = train_df['EnergyRatingCat']


# In[68]:

cleaned_train_df1[rel_features].isnull().sum()


# In[69]:

cleaned_train_df1.head()


# In[70]:

cleaned_train_df1 = cleaned_train_df1[rel_features]


# In[71]:

cleaned_train_df1.isnull().sum()


# In[72]:

cleaned_train_df1['AvgWallU'].fillna(cleaned_train_df1['AvgWallU'].mean(), inplace = True)


# In[73]:

cleaned_train_df1 = cleaned_train_df1.dropna()


# In[74]:

plt.hist(cleaned_train_df1['EnergyRatingCont'],bins =1000)
plt.xlim(0,1000)


# In[75]:

removed_outliers = cleaned_train_df1[cleaned_train_df1['EnergyRatingCont'] < 800]


# In[76]:

len(train_df), len(cleaned_train_df), len(cleaned_train_df1)


# In[77]:

cleaned_train_df1 = cleaned_train_df1.set_index('BuildingID')


# In[78]:

cleaned_train_df1.head()


# In[79]:

cleaned_train_df1.dtypes


# In[80]:

features = rel_features


# In[81]:

features[1:-2]


# In[82]:

y = cleaned_train_df1['EnergyRatingCat']
x = cleaned_train_df1[features[1:-2]]


# ## Training Decision Tree

# In[83]:

from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[84]:

dt = DecisionTreeClassifier()


# In[85]:

dt.fit(x,y)


# In[86]:

dt.score(x,y)


# ## Training Random Forest 

# In[87]:

from sklearn.ensemble import RandomForestClassifier


# In[88]:

rf = RandomForestClassifier(n_estimators=10)


# In[89]:

rf.fit(x,y)


# In[90]:

rf.score(x,y)


# ## Training Neural Network

# In[91]:

from sklearn.neural_network import MLPClassifier


# In[92]:

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# In[93]:

nn.fit(x,y)


# In[94]:

nn.score(x,y)


# ### Importing test data 

# In[95]:

test_df = pd.read_csv("data/testing.txt", sep = '\t', encoding = 'iso-8859-1')


# In[96]:

test_df.head()


# In[97]:

test_df['Year'] = pd.to_numeric(test_df['Year'].str.replace('Y',''), errors = 'coerce')
test_df['GroundFloorArea'] = pd.to_numeric(test_df['GroundFloorArea'].str.replace('sq. m',''), errors = 'coerce')
num_test_df = test_df.select_dtypes(include = numerics)
cleaned_test_df = num_test_df.dropna(axis = 1, thresh = 100000)
cleaned_test_df.fillna(cleaned_test_df.mean(), inplace = True)


# In[98]:

cleaned_test_df1 = cleaned_test_df[features[1:-2]]
cleaned_test_df1.head()


# In[99]:

x_test = cleaned_test_df1


# In[100]:

out = dt.predict(x_test)


# In[101]:

len(out), len(x_test)


# In[102]:

out


# In[103]:

cleaned_test_df['EnergyRatingCat']= out


# In[104]:

test_groups = cleaned_test_df.groupby("EnergyRatingCat")['EnergyRatingCat'].count()


# In[105]:

train_groups = train_df.groupby('EnergyRatingCat')['EnergyRatingCat'].count()


# In[106]:

cleaned_test_df.groupby("EnergyRatingCat")['EnergyRatingCat'].count().plot(kind = 'bar')


# In[107]:

train_df.groupby('EnergyRatingCat')['EnergyRatingCat'].count().plot(kind = 'bar')


# In[108]:

(test_groups/test_groups.max()).plot(kind = 'bar')


# In[109]:

(train_groups/train_groups.max()).plot(kind = 'bar')


# In[110]:

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
width = 0.4
(test_groups/test_groups.max()).plot(kind = 'bar', ax = ax, alpha = 0.5, color = 'red', position = 0, width = width)
(train_groups/train_groups.max()).plot(kind = 'bar', ax = ax2, alpha = 0.5, color = 'blue', position = 1, width = width)


# In[111]:

from sklearn.model_selection import cross_val_score


# In[126]:

cross_val_score(dt, x, y, cv = 3)


# In[114]:

cross_val_score(rf, x, y, cv = 3 )


# In[116]:

from sklearn.model_selection import cross_val_predict


# In[117]:

y_train_pred = cross_val_predict(rf, x, y, cv =3)


# In[121]:

from sklearn.metrics import confusion_matrix


# In[123]:

conf_mx = confusion_matrix(y, y_train_pred)


# In[124]:

plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()


# Need to investigate the source of the errors. Perhaps need to clean the data more and add more features, and actually utilise the categorical fields. 

# In[ ]:



