import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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