import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from joblib import dump,load


data_raw=pd.read_csv(r'C:\Users\Mohit\Desktop\web\ML\ML_Project_1\ML Project 1\housing_data.csv')
# data_raw.hist(bins=50, figsize=(20, 15))
# plt.show()

# print(data_raw.info())
#print(data_raw.head())

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data_raw,data_raw['CHAS']):
    strat_train_set=data_raw.loc[train_index]
    strat_test_set=data_raw.loc[test_index]

my_pipeline=Pipeline(
    [('imputer',SimpleImputer(strategy="median")),
     ('std_scaler',StandardScaler())]
)

strat_train_set_temp=strat_train_set.drop('MEDV',axis=1)

some_data=strat_train_set_temp.iloc[:5]
housing_tr=my_pipeline.fit_transform(strat_train_set_temp)
housing_labels=strat_train_set['MEDV'].copy()
some_labels=housing_labels.iloc[:5]

prepared_dt=my_pipeline.transform(some_data)
model=LinearRegression()

model.fit(housing_tr,housing_labels)
predicted_labels=model.predict(prepared_dt)
print(list(some_labels),predicted_labels)


scores=cross_val_score(model,housing_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


#print_scores(rmse_scores)




