
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[18]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
dataframe.head()


# In[10]:


#Let's convert the DataFrame object to a NumPy array to achieve faster computation. Also, 
#let's segregate the data into separate variables so that the features and the labels are separated.
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


# In[14]:


#First Method - Select K best using chi square
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[16]:


# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
#SO this score  shows  the scores against each variable


# In[19]:


features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])
#So when we see actual data we can see this 4 variable  belongs to plas, test, mass, and age which are top 4 varibles


# In[31]:


#Now instead of creating an array we can also directly do on actual  data
X1 = dataframe.ix[:,0:8]
Y1 = dataframe.ix[:,8]


# In[40]:


test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X1, Y1)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X1)
# Summarize selected features
print(features[0:5,:])


# In[42]:


#Second method is RFE - Recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#You will use RFE with the Logistic Regression classifier to select the top 3 features. 
#The choice of algorithm does not matter too much as long as it is skillful and consistent.


# In[43]:


# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
#You can see that RFE chose the top 3 features as preg, mass, and ped


# In[47]:


#Third is Ridge regression which is basically a regularization technique 
#and an embedded feature selection techniques as well.
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)


# In[48]:


# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


# In[49]:


#Next, you will pass Ridge model's coefficient terms to this little function and see what happens.
print ("Ridge model:", pretty_print_coefs(ridge.coef_))
#You can spot all the coefficient terms appended with the feature variables.


# In[55]:


# Fourt PCA - Feature Extraction with PCA
from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


# In[65]:


#Fifth Feature Importance
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
#You can see that we are given an importance score for each attribute where the larger score the more important the attribute.
#The scores suggest at the importance of plas, age and mass.
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()

pd.Series(model.feature_importances_, index=X1.columns).nlargest(4).plot(kind='barh') 


# In[80]:


#Correlation
corr_matrix = X1.corr()
corr_matrix
#plt.imshow(corr_matrix, cmap = "coolwarm")


# In[82]:


plt.matshow(X1.corr())
plt.xticks(range(len(X1.columns)), X1.columns)
plt.yticks(range(len(X1.columns)), X1.columns)
plt.colorbar()
plt.show()


# In[99]:


# Find correlation and remove variables which  are highly correlated
corr_matrix = X1.corr().abs()
corr_matrix
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper
# Find index of feature columns with correlation greater than 0.5
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
to_drop
# Drop features 
X1.drop(to_drop,axis = 1)


# In[100]:


#Near Zero Variance
X1 = dataframe.ix[:,0:8]
Y1 = dataframe.ix[:,8]
X1.columns


# In[123]:


# Create VarianceThreshold object with a variance with a threshold of 0.5
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=.5)
# Conduct variance thresholding
X_high_variance = thresholder.fit_transform(X1)
X_high_variance[0:2]


# In[131]:


# Information  Value / Weight of evidence
#final_iv, IV = data_vars(X1 , X2)
# No direct package in python we can use excel
import os
os.getcwd()

