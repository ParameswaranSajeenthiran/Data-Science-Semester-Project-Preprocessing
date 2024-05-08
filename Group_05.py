import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# Load the data
employees = pd.read_csv('employees.csv')
# print(employees.head())
pd.options.mode.chained_assignment = None  # default='warn'

# Check the data types
print("preprocessing ....")
# print("Data types:")
# print(employees.dtypes)

# Check for missing values
# print(employees.isna().sum())





employees=employees.replace("'0000'", 0)



####################################### imputing  Year of Birth using KNN imputer ######################################
employees['Year_of_Birth']=(employees['Year_of_Birth']).astype(int)
employees=employees.replace(0, np.nan)
gender={
    'Male':True,
    'Female':False
}
status={
     'Active':True,
     'Inactive':False
 }
employees['Gender'] = employees['Gender'].map(gender)
employees['Status'] = employees['Status'].map(status)

for colname in employees.select_dtypes(["object", "category"]):
    employees[colname], _ = employees[colname].factorize()

marital_nap = {
    0: False,
    1: True,
    -1:np.nan
}
#encoding Marital_Status
employees['Marital_Status'] = employees['Marital_Status'].map(marital_nap)
#selecting required features for imputing
employess_=employees[['Gender','Religion_ID','Marital_Status','Designation_ID','Status','Employment_Category','Employment_Type','Designation','Year_of_Birth']]
imputer = KNNImputer(n_neighbors=2)
After_imputation = imputer.fit_transform(employess_)

data=pd.DataFrame(After_imputation)

employees = pd.read_csv('employees.csv')
employees['Year_of_Birth']=data[8]



################################################ imptuting marital status ##############################################
employees = pd.read_csv('employees.csv')
employess_=employees[['Gender','Religion_ID','Marital_Status','Designation_ID','Status','Employment_Category','Employment_Type','Designation','Year_of_Birth']]
X_test=  employess_.loc[(employess_['Marital_Status'] !='Married') &  (employess_['Marital_Status'] !='Single')]
X_train=  employess_.loc[(employess_['Marital_Status'] =='Married') |  (employess_['Marital_Status'] =='Single')]
y_train=X_train.pop('Marital_Status')
y_test=X_test.pop('Marital_Status')
X_train['Year_of_Birth']=data[8]
X_test['Year_of_Birth']=data[8]

for colname in X_train.select_dtypes(["object", "category"]):
    X_train[colname], _ = X_train[colname].factorize()

for colname in X_test.select_dtypes(["object", "category"]):
    X_test[colname], _ = X_test[colname].factorize()

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=None)
tree_clf.fit(X_train, y_train)
predcitions=tree_clf.predict(X_test)
X_test['Marital_Status']=predcitions

for index, row in X_test.iterrows():
  employees.at[index, 'Marital_Status']=row['Marital_Status']

employees['Year_of_Birth']=data[8]
X_train['Marital_Status']=y_train
employees=employees.drop(['Reporting_emp_2' ,'Name' , 'Title' ,'Religion_ID' ,'Designation_ID'],axis=1)


###################### extracting data from leave csv files and merging with employees.csv #############################
leaves = pd.read_csv('leaves.csv')
unique_leaves = leaves[['Employee_No', 'leave_date', 'Type']].drop_duplicates()

# Group the unique leaves dataframe by 'Employee_No'
grouped_leaves = unique_leaves.groupby('Employee_No')

# Create a new dataframe with the counts of Half Day and Full Day values for each employee
leave_counts = grouped_leaves['Type'].value_counts().unstack(fill_value=0)

# Rename the columns for clarity
leave_counts.columns = ['Half Day', 'Full Day']

employees=employees.set_index("Employee_No")
employees=employees.assign(No_Of_Half_Day_Leaves=0)
employees=employees.assign(No_Of_Full_Day_Leaves=0)
for index, row in leave_counts.iterrows():
  # print(index)
  employees.at[index, 'No_Of_Half_Day_Leaves']=row['Half Day']
  employees.at[index, 'No_Of_Full_Day_Leaves']=row['Full Day']


# handling missing Employee_No in employees.csv
employees=employees[:997]

################################# handling data duplicity in Date_Resgined and Inactive ################################

mask = (employees['Status'] == 'Inactive') & (employees['Date_Resigned'].isin(['\\N', '0000-00-00'])) & (~employees['Inactive_Date'].isin(['\\N', '0000-00-00']))
employees.loc[mask, 'Date_Resigned'] = employees.loc[mask, 'Inactive_Date']

mask=employees.loc[(employees['Date_Resigned'].isin([ '0000-00-00'])) & (employees['Status'] == 'Active')]
for index, row in mask.iterrows():
  employees.at[index, "Date_Resigned"] = "\\N"

employees.rename(columns = {'Date_Resigned':'Date_Resigned/Inactive'}, inplace = True)
employees=employees.replace("'0000'", 0)
employees=employees.drop(['Inactive_Date'], axis=1)


employees['Year_of_Birth'] = employees['Year_of_Birth'].astype(int)
employees['No_Of_Half_Day_Leaves'] = employees['No_Of_Half_Day_Leaves'].astype(int)
employees['No_Of_Full_Day_Leaves'] = employees['No_Of_Full_Day_Leaves'].astype(int)




################################ extracting data from salary csv files and merging with employees.csv ##################

employees_raw = pd.read_csv('employees.csv')
unique_employees = employees_raw['Employee_No'].unique()
salary = pd.read_csv('salary.csv' )
sal = salary[salary['Employee_No'].isin(unique_employees)]
grouped=sal.groupby('Employee_No')

# introduce new columns in employees.csv
for col in [ 'Net Salary', 'Total Working Days', 'OT Hours', 'Total Deduction', 'Allow - Extra Working Hours'	]:
  employees[col]=np.nan

#finding ooutliers  and replacing them with mean
def find_outliers_iqr(data):
    # Calculate first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate interquartile range (IQR)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    # print( "No Of Outliers :"+ str(len(outliers)))
    # print(outliers)

    return outliers

for key, item in grouped:
    # print('.', end="")

    df=pd.DataFrame(grouped.get_group(key))
    for col in df :
        if key== 349:
            pass
            # print(df[col])

        if col in [ 'Net Salary', 'Total Working Days', 'OT Hours', 'Total Deduction', 'Allow - Extra Working Hours'	]:

          outliers = find_outliers_iqr(df[col])

            # Replace outliers with the mean of other rows
          non_outliers_mean = df[col][~df[col].isin(outliers)].mean()
          df.loc[outliers.index, col] = non_outliers_mean

          if(df[col].mean()==np.nan):
            print(df[col])

          employees.at[key, col]=df[col].mean()




#################### extracting data from attendance csv files and merging with employees.csv ##########################
employees_raw = pd.read_csv('employees.csv')
attendance = pd.read_csv('attendance.csv' ,low_memory=False)
unique_employees = employees_raw['Employee_No'].unique()
attendance = attendance[attendance['Employee_No'].isin(unique_employees)]
mode_shift_start = attendance.groupby('project_code')['Shift_Start'].agg(pd.Series.mode)
attendance.loc[attendance['Shift_Start'] == '0:00:00', 'Shift_Start'] = attendance['project_code'].map(mode_shift_start)
attendance['time'] = (pd.to_datetime(attendance['in_time']) - pd.to_datetime(attendance['Shift_Start'])).dt.total_seconds() / 60

p = attendance[(attendance['time'] <240) & (attendance['time'] > (-180))]
p['Employee_No'].nunique()
p.loc[p['time'] < 0, 'time'] = 0

employee_time_df = p[['Employee_No', 'time']]

employee_time_df['time'] = employee_time_df.groupby('Employee_No')['time'].transform('mean')
employees=employees.assign(Average_Late_Time=0)
employees['Average_Late_Time'] = employee_time_df['time']


################### Imputing missing in the salary data and Average_Late_Time ###########################################
employees_copy=employees.copy()
for colname in employees.select_dtypes(["object", "category"]):
    employees_copy[colname], _ = employees_copy[colname].factorize()
# fit regression model using Bayesian Ridge
imputer = IterativeImputer( )
imputed = imputer.fit_transform(employees_copy)
employees=employees.reset_index()

employees_copy_immputed = pd.DataFrame(imputed, columns=employees_copy.columns)
for col in [ 'Net Salary', 'Total Working Days', 'OT Hours', 'Total Deduction', 'Allow - Extra Working Hours', 'Average_Late_Time']:
    employees[col]=employees_copy_immputed[col]
employees.to_csv('employee_preprocess_Group_05.csv',index=False)
print("Done")