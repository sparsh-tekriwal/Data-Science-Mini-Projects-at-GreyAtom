# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset= pd.read_csv(path)


# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
del(dataset["Id"])
# check the statistical description
print(dataset.describe())



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols=dataset.columns

#number of attributes (exclude target)
size=len(cols)-1

#x-axis has target attribute to distinguish between classes
x=dataset.drop(cols[-1], 1)

#y-axis shows values of an attribute
y=dataset[cols[-1]]

#Plot violin for all attributes



# --------------
import numpy as np
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'

subset_train = dataset.iloc[:,:num_features]
cols = subset_train.columns

#Calculate the pearson co-efficient for all possible combinations
data_corr = subset_train.corr()
#sn.heatmap(data_corr)
data_corr = np.abs(data_corr)
data_cols = data_corr.columns
threshold = 0.5
corr_var_list = []
for i in np.arange(0,data_corr.shape[0]):
  for j in np.arange(0,i+1):
      if((data_corr.iloc[i,j]>=threshold) & (data_corr.iloc[i,j]<1)):
          corr_var_list.append(data_corr.iloc[i,j])
      #corr_var_list
s_corr_list =sorted(corr_var_list,reverse=True)


# --------------
#Import libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X= dataset.drop(["Cover_Type"],1)
Y= dataset.Cover_Type

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, random_state=0, test_size=0.2)
ss=StandardScaler()
X_train_temp=pd.DataFrame(ss.fit_transform(X_train))
X_test_temp=pd.DataFrame(ss.transform(X_test))

X_train1=pd.concat([X_test_temp, X_train], axis=1)


#Standardized
#Apply transform only for non-categorical data



#Concatenate non-categorical data and categorical

 




# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

skb= SelectPercentile(score_func= f_classif, percentile=20)
predictors= skb.fit_transform(X_train, Y_train)
print(predictors)
scores= skb.scores_ 


top_k_index= sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:predictors.shape[1]]

top_k_predictors= [X_train.columns[i] for i in top_k_index]



# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features=clf1.fit(X_train, Y_train)
predictions_all_features= clf1.predict(X_test)
score_all_features= accuracy_score(Y_test, predictions_all_features)

model_fit_top_features= clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features= clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features= accuracy_score(Y_test, predictions_top_features)


