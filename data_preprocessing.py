import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#importing the dataset and separating features and labels
dataset = pd.read_csv("data/Machine-Learning-A-Z-Codes-Datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
print(dataset)

#data to be trained on
x = dataset.iloc[: , :-1].values
print(x)

#data to be predicted
y = dataset.iloc[: , -1].values
print(y)

#missing data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

#encoding categorical data
ct = ColumnTransformer(transformers = [("encoder" , OneHotEncoder() , [0])] , remainder = "passthrough")
x = np.array(ct.fit_transform(x))

print(x)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#train validate test split
x_train , x_temp , y_train , y_temp = train_test_split(x , y , test_size = 0.4 , random_state = 1)

x_validation , x_test , y_validation , y_test = train_test_split(x , y , test_size = 0.5 , random_state = 1)


print(x_train)
print(y_train)
print(x_validation)
print(y_validation)
print(x_test)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_validation = sc.transform(x_validation)
x_test = sc.transform(x_test)
