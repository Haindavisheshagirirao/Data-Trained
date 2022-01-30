#!/usr/bin/env python
# coding: utf-8

# Goal: Here our goal is to predict the employee attrition:

# Description: "Attrition" is a major problem for every organization. Here we predict that how HR Analytics analyze the employee
#              attrition?

# # Importing the libraries:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import  train_test_split, cross_val_score
#from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import LinearSVC , SVC

from sklearn.ensemble import  RandomForestClassifier , GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore") 


# # Data Collection:

# In[2]:


Attrition_data = pd.read_csv("HR-Employee.csv")


# In[3]:


Attrition_data.head(7)


# # What are the Columns?

# In[4]:


Attrition_data.columns


# In[5]:


# Observation: 1) Here we can see that the dataset consists of the general data of the employee , professional data of the employe ,
# personal data of the employee and are combined to analyze the attrition of the employee.
#              2) Here our target column is "Attrition".


# # Size of the data:

# In[6]:


Attrition_data.shape


# In[7]:


# Observation: Here we can see that there are : Rows - 1470 , Columns - 35.


# # What are the different datatypes present?

# In[8]:


Attrition_data.dtypes


# In[9]:


# Observation: Here we can see that the major columns are with the datatype: "int64" and there are few columns with the "object"
# datatype.


# # Information of the data:

# In[10]:


Attrition_data.info()


# In[11]:


# Observation: Here we can see that there are no null values in any of the columns.


# # Statistical Analysis of the data:

# In[12]:


Attrition_data.describe()


# In[13]:


# Observation: 1) Here we can see that the numerical columns are considered. we can see that these columns are with absolute count.
# 2) Also we can see that the difference between "mean and std" is more in many of the columns but in few columns it is less.
# 3) Also in few columns the minimum value is "0".


# # Checking with the null values in the data:

# In[14]:


Attrition_data.isnull().sum()


# In[15]:


# Observation: here we can see that this code checked the complete null values in all the columns including object datatype 
# columns and we can see that no null values in any of the columns.


# In[16]:


# Also can the null/missing values:

Attrition_data.isnull().values.any()


# # Checking the Count of our label: "Attrition" :

# In[17]:


Attrition_count = pd.DataFrame(Attrition_data["Attrition"].value_counts())


# In[18]:


Attrition_count


# In[19]:


# Observation: Here we can see that : The employees attritioned - 237
#                                     The employees who have not attritioned - 1233.


# # Removing the unrequired columns:

# In[20]:


Attrition_data = Attrition_data.drop(["EmployeeCount" , "EmployeeNumber" , "Over18", "StandardHours"] , axis = 1)


# In[21]:


Attrition_data.shape


# In[22]:


# Observation: here we can see that the number of columns are reduced to 32,that means we have successfully dropped our unrequired columns.


# # Visualization:

# ## Univariate Analysis:

# ### Plot of our target variable : "Attrition"

# In[23]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.Attrition);


# In[24]:


# Observation : Here we can see that there are employees with attrition - yes in "blue" color and there are employees with
# attrition - no in the othet color.


# In[25]:


Attrition_data.columns


# # Age:

# In[26]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['Age'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['Age']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['Age']);


# In[27]:


# Observation: Here we can see that the graph is almost distributed normally and also we can see that there are no outliers 
# detected.


# In[28]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['Age'], kde=False);#, #bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['Age']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['Age']);


# # BusinessTravel:

# In[29]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.BusinessTravel);


# In[30]:


# Observation: Here we can see that there are more number of employees who "Travel_Rarely" and the order is followed by, who
# "Travel frequently Bussiness travel" and finally by, who "Non - Travel" ie., who dont travel.


# In[31]:


Attrition_data.head()


# # DailyRate:

# In[32]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['DailyRate'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['DailyRate']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['DailyRate']);


# In[33]:


# Observation: here we can see that the distribution curve is broad and seems to be distributed almsot normally and also we can
# see that there are no outliers in the boxplot.


# # Department:

# In[34]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.Department);


# In[35]:


# Observation: Here we can see that "research and Development" bar is higher than the other bars and least one is "Human Resources"


# # DistanceFromHome:

# In[36]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['DistanceFromHome'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['DistanceFromHome']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['DistanceFromHome']);


# In[37]:


# Observation: Here we can see that there no outliers present in the boxplot but the distribution curve is not at all normall and
# is skewed towards right and so it is right skewed due to which boxplot is onesided.


# # Education:

# In[38]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.Education);


# In[39]:


# observation: here we can see that the high count of the education is seen in "Category - 3" and the least is seen in "category - 5"


# # EducationField:

# In[40]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.EducationField);


# In[41]:


# Observation: here we can see that there is highest count for the bar "Life sciences" and least one is "Human Resources".


# # EnvironmentSatisfaction:

# In[42]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.EnvironmentSatisfaction);


# In[43]:


# Observation: Here we can see that there is high count in the "category - 3" and also we can see that category 3 & 4 are almost
# same.


# # Gender:

# In[44]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.Gender);


# In[45]:


# Observation: Here we can see that the high count is for "male" category than "Female".


# # HourlyRate:

# In[46]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['HourlyRate'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['HourlyRate']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['HourlyRate']);


# In[47]:


# Observation: Here we can see that there are no outliers can be in the boxplot and the distribution curve is almost "normal".


# # JobInvolvement:

# In[48]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.JobInvolvement);


# In[49]:


# Observation: Here we can see that the highest count can be seen in "Category - 3" and least in "category - 1". 


# # JobLevel:

# In[50]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['JobLevel'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['JobLevel']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['JobLevel']);


# In[51]:


# Observation: Here we can see that there no outliers in the boxplot but also it is onesided and the distribution is not at all normal


# # JobRole:

# In[52]:


plt.figure(figsize=(20,5))
sns.countplot(Attrition_data.JobRole);


# In[53]:


# Observation: here we can see that the high count is for the category "sales Executive"  and the least is for the "Human Resiurces".


# # JobSatisfaction:

# In[54]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.JobSatisfaction);


# In[55]:


# Observation: Here we cna see that there are is similar to the "environment satisfaction" column and the category 3 & 4 are 
# almost similar in their count and the least one is "category - 2".


# # MaritalStatus:

# In[56]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.MaritalStatus);


# In[57]:


# Observation: Here we can see that there are more number of employees who are married and the least one are divorced.


# # MonthlyIncome:

# In[58]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['MonthlyIncome'], kde=False, bins=range(800, 10000, 1));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['MonthlyIncome']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['MonthlyIncome']);


# In[59]:


# Observation: here we can see that there no outliers present in the boxplot and the distribution is not at all normal.


# # MonthlyRate:

# In[60]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['MonthlyRate'], kde=False, bins=range(500, 10000, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['MonthlyRate']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['MonthlyRate']);


# In[61]:


# Observation: Here we can see that the boxplot has no putliers and the distribution curve is broad in the middle indicating to
# be almost normal.


# # NumCompaniesWorked:

# In[62]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['NumCompaniesWorked'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['NumCompaniesWorked']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['NumCompaniesWorked']);


# In[63]:


# Observation : Here we can see that there is an outlier present which can be seen in boxplot but is very far from "max - quantile"
# point and so mostly our model will not be affected by it.


# # OverTime:

# In[64]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.OverTime);


# In[65]:


#  Observation: Here we can see that the "Category - No" is high when compared to "Yes". 


# # PercentSalaryHike:

# In[66]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['PercentSalaryHike'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['PercentSalaryHike']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['PercentSalaryHike']);


# In[67]:


# Observation: Here we can see that the boxplot is with no outliers and the distribution curve is skewed towards right and is 
# not normal.


# # PerformanceRating:

# In[68]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.PerformanceRating);


# In[69]:


# observattion: Here we can see that the more number of employees are with performance rating - 3 and very few with performace
# rating - 4.


# # RelationshipSatisfaction:

# In[70]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.RelationshipSatisfaction);


# In[71]:


# Observation: here we can see that the relationship satisfaction rate is high in category - 3 and least in category - 1.


# # StockOptionLevel:

# In[72]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.StockOptionLevel);


# In[73]:


# observation: Here we can see that the level is more in category - 0 and least in category - 3.


# # TotalWorkingYears:

# In[74]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['TotalWorkingYears'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['TotalWorkingYears']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['TotalWorkingYears']);


# In[75]:


# Observation: Here we can see that there are outliers present in the boxplot at the max - quantile andthe distribution is 
# somewhat skewed.


# # TrainingTimesLastYear:

# In[76]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.TrainingTimesLastYear);


# In[77]:


# Observation: Here we can see that there are more number of employees who got trained 2 times last year and there are few 
# employees who did not get trained last year.


# # WorkLifeBalance:

# In[78]:


plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.WorkLifeBalance);


# In[79]:


# Observation: Here we can see that there is more count in category - 3  and least in category - 1 for Worklifebalance.


# # YearsAtCompany:

# In[80]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['YearsAtCompany'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['YearsAtCompany']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['YearsAtCompany']);


# In[81]:


# Observation: Here we can see that there are outliers present in the boxplot at the max - quantile andthe distribution is 
# somewhat skewed.


# # YearsInCurrentRole:

# In[82]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['Age'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['Age']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['Age']);
plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.YearsInCurrentRole);


# In[83]:


# Observation: here we can see that there are no outliers present in the boxplot and the distribution curve seems to be normal
# and probably more number of employees are with 2 years of experience.


# # YearsSinceLastPromotion:

# In[84]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['YearsSinceLastPromotion'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['YearsSinceLastPromotion']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['YearsSinceLastPromotion']);
plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.YearsSinceLastPromotion);


# In[85]:


# Observation: here we can see that there are outliers present in the boxplot and are far away from eachother and the distribution
# curve is not at all normal and there are more number of employees with no experience after getting promotion ore recently 
# promoted.


# # YearsWithCurrManager:

# In[86]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.distplot(Attrition_data['YearsWithCurrManager'], kde=False, bins=range(0, 31, 2));
plt.subplot(2,2,2)
sns.boxplot(Attrition_data['YearsWithCurrManager']);
plt.subplot(2,2,3)
sns.distplot(Attrition_data['YearsWithCurrManager']);
plt.figure(figsize=(5,5))
sns.countplot(Attrition_data.YearsWithCurrManager);


# In[87]:


# Observation: here we can see that thare are few outliers present in the boxplot and the distribution curve is with 2 peaks and
# is not at all normal and also we can see that there are more number of employees with 2 years of experience with the current 
# manager.


# # Multivariate Analysis:

# ## correlation:

# In[88]:


plt.figure(figsize = (14,15))
sns.heatmap(Attrition_data.corr(),cbar = True, square = True, fmt = ".0%", annot = True, annot_kws = {'size': 8}, cmap = 'Blues')


# In[89]:


# Observation: 1) Here we can see that there are columns with high correlation or positive correlation with the other columns like: "Years at 
# company" with "joblevel", "Monthly income" also "Years at company" with "Totalworking years","years at currentrole", "Years sice
# last promotion", "Years with current manager".
#              2) Here we can see that there are columns with positive correlation with the column "Years in current role" which
# is with "Years at Company","Years since last promotion","years with current manager".
#              3) Here we can see that there are columns with positive correlation with the column "years since last promotion"
# which is with "Years at company".
#              4) Also we can see that the highest of the positive correlation value is in the column "Monthly income" which is
# with "joblevel".


# # Transforming the data:

# ## Transforming the non-numerical columns into numerical:

# In[90]:


from sklearn.preprocessing import LabelEncoder


# In[91]:


# Inspite of mentioning each and every which has strings we user "forloop" here as the data seems to be big.


# In[92]:


for column in Attrition_data.columns:
    if Attrition_data[column].dtype == np.number:
        continue
    Attrition_data[column] = LabelEncoder().fit_transform(Attrition_data[column])
        


# In[93]:


Attrition_data


# In[94]:


# Observation: here we can see that all the non-numerical columns are converted into numerical columns.


# # Checking the Outliers:

# In[95]:


col_list = Attrition_data.columns.values
ncol = 32
nrows = 12
plt.figure(figsize = (ncol,5*ncol))
for i in range (0, len(col_list)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(data = Attrition_data[col_list[i]],color = 'green', orient = 'v')
    plt.tight_layout()


# In[96]:


# Observation: Here we can see that there are columns with the many number of the outliers are present in the columns - 
# 1)"YearsAtCompany", 2) "MonthlyIncome"  3) "TotalWorkingYears"  4)"YearsSinceLastPromotion"  5)"YearsWithCurrManager".


# # Removing Outliers -  Using Z- Score method:

# In[97]:


from scipy.stats import zscore
z = np.abs(zscore(Attrition_data))
z.shape


# In[98]:


threshold = 3
print(np.where(z>3))


# In[99]:


len(np.where(z>3))


# In[100]:


len(np.where(z>3)[0])


# In[101]:


Attrition_data_new = Attrition_data[(z<3).all(axis = 1)]
print(Attrition_data.shape)
print(Attrition_data_new.shape)


# In[102]:


# observation: here we can see that there is reduce in the number of total records(rows) and so there is reduce in the data.


# # loss percentage calculation:

# In[103]:


loss_percent = (1470-1387)/1470*100
print(loss_percent)


# In[104]:


# Observation: Here we can see that there is loss of 5.6% of the data , so it is negligible.


# # Skewness:

# In[105]:


Attrition_data_new.skew()


# In[106]:


features = ["YearsAtCompany","TotalWorkingYears","YearsSinceLastPromotion","YearsWithCurrManager"]


# In[107]:


from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')
'''
parameters:
method = 'box-cox' or 'yeo-johnson'
'''


# In[108]:


Attrition_data_new[features] = scaler.fit_transform(Attrition_data_new[features].values)
Attrition_data_new[features]


# In[109]:


Attrition_data_new.skew()


# In[110]:


# Observation: Here we can see that there is change or we can say that reduce in the skewness of the data.


# # Data Preprocessing:

# ## Separating independent and the target variables:

# ### train_test_split:

# In[111]:


x = Attrition_data_new.drop("Attrition", axis=1)
y = Attrition_data_new["Attrition"]


# In[112]:


x


# In[113]:


y


# ### Scaling the x_data using standardscaler:

# In[114]:


scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
x


# In[115]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# # Training the model:

# In[116]:


# here we will use "forloop" for using models continously without getting interrupted:


# In[117]:


models = {"LogisticRegression" : LogisticRegression(),
          "K-Nearest Neighbors": KNeighborsClassifier(),
          "Decision Tree"      : DecisionTreeClassifier(),
          "Random Forest"      : RandomForestClassifier(),
          "Gradient Boosting"  : GradientBoostingClassifier()}

for name, model in models.items():
    model.fit(x_train,y_train)
    print(name + " is trained now.")


# # testing:

# In[118]:


for name, model in models.items():
    print(name + ": {:,.2f}%".format(model.score(x_test,y_test)*100))


# # HyperParameter tuning:

# ## GridSearchCV:

# In[119]:


from sklearn.model_selection import GridSearchCV


# In[120]:


parameters = {'criterion':['mse', 'mae'],
             'max_features':['auto', 'sqrt', 'log2'],
             'n_estimators':[0,20],
             'max_depth':[2,4,6]}


# # GradientBoostingClassifier:

# In[121]:


# here we use "GradientBoostingClassifier" because this model has highest accuracy score when compared to the other models.


# In[122]:


GCV=GridSearchCV(GradientBoostingClassifier(),parameters,cv=5)


# In[123]:


GCV.fit(x_train,y_train)


# In[124]:


GCV.best_params_


# In[125]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[126]:


parameters = {'criterion': ['mse','mae'], 'max_features': ['auto', 'sqrt', 'log2']}

GradientBoosting = GradientBoostingClassifier()
Classifier = GridSearchCV(GradientBoosting,parameters)
Classifier.fit(x_train,y_train)

print(Classifier.best_params_)


# In[127]:


# observation: here we can see that the best_parameteers are selected.


# In[128]:


GradientBoosting = GradientBoostingClassifier(criterion = 'mse', max_features = 'log2')
GradientBoosting.fit(x_train,y_train)
GradientBoosting.score(x_train,y_train)

pred_decision = GradientBoosting.predict(x_test)
GradientBoostingS = r2_score(y_test,pred_decision)
print('R2 Score:', GradientBoostingS*100)

GradientBoostingScore = cross_val_score(GradientBoosting,x,y,cv = 5)
GradientBoostingC = GradientBoostingScore.mean()
print("Cross Val Score:",GradientBoostingC*100)


# In[129]:


# Observation: here we can see that the best parameters we got for the model.


# In[130]:


GradientBoosting = GradientBoostingClassifier(criterion = 'mse', max_features = 'sqrt')
GradientBoosting.fit(x_train,y_train)
GradientBoosting.score(x_train,y_train)

pred_decision = GradientBoosting.predict(x_test)
GradientBoostingS = r2_score(y_test,pred_decision)
print('R2 Score:', GradientBoostingS*100)

GradientBoostingScore = cross_val_score(GradientBoosting,x,y,cv = 5)
GradientBoostingC = GradientBoostingScore.mean()
print("Cross Val Score:",GradientBoostingC*100)


# # saving the model:

# In[131]:


import pickle
filename = 'churn.pkl'
pickle.dump(GradientBoosting,open(filename, 'wb'))


# In[132]:


loaded_model = pickle.load(open("churn.pkl", "rb"))
result = loaded_model.score(x_test, y_test)
print(result)


# # conclusion:

# In[133]:


conclusion = pd.DataFrame([loaded_model.predict(x_test)[:],pred_decision[:]],index = ["Predicted","Original"])


# In[134]:


conclusion


# So, therefore our best model is "GradientBoosting" and the percentage we achieved is 88% .

# In[ ]:




