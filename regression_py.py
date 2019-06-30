mport csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#this is a test program

output = {1: 'Positive' , 0 : 'Negative'}       

#Reading csv file
df = pd.read_csv('../Downloads/diabetes.csv') 

#dropping the columns
df1 = df.drop(columns = ['preg'])         

# Separating X and Y
X = df1[df1.columns[0 : 7]]
to_encode = df1[df1.columns[7]]

# label Encoding 
from sklearn.preprocessing import LabelEncoder
Encode = LabelEncoder().fit_transform(to_encode)
Y = pd.Series(Encode)

#Calucalting the % of data to split
row_count = df1.shape[0]
split = int(0.7 * row_count)

#Splitting to tain and test
x_train = X[ : split]
x_test = X[split : ]
y_train = Y[ : split]
y_test = Y[split : ]
#print(x_test)

#Logistic Regression
log_reg = LogisticRegression().fit(x_train, y_train)
print('{0} coeff {1} intercept'.format(log_reg.coef_, log_reg.intercept_))
predictions = log_reg.predict(x_test)
P = pd.Series(predictions)
print(predictions)

#Checking the accuracy
from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test, P)
print(c)
# Accuracy = 79.65
