from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

train_set_raw = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\titanic\train.csv')
test_Set = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\titanic\test.csv')

x_test = test_Set.drop(['Ticket'], axis=1).copy()
x_test = x_test.drop(['Name'], axis=1).copy()

pt_train = train_set_raw.corr()

# sn.heatmap(pt_train)
# plt.show()


y_data_tr = train_set_raw['Survived']
x_data_trr = train_set_raw.drop(['Survived'], axis=1).copy()
x_data_trn = x_data_trr.drop(['Ticket'], axis=1).copy()
x_data_tr = x_data_trn.drop(['Name'], axis=1).copy()

cat_cols = [col for col in x_data_tr.columns if x_data_tr[col].dtype == 'object' and x_data_tr[col].nunique() < 10]
num_cols = [col for col in x_data_tr.columns if
            x_data_tr[col].dtype == 'int64' or train_set_raw[col].dtype == 'float64']


x_train, x_val, y_train, y_val = train_test_split(x_data_tr, y_data_tr, test_size=0.2, train_size=0.8, random_state=0)

num_imputer = SimpleImputer(strategy='most_frequent')
cat_one_hot = OneHotEncoder(handle_unknown='ignore')

cols_preprocessor = ColumnTransformer(transformers=[
    ('num', num_imputer, num_cols),
    ('cat', cat_one_hot, cat_cols)
])

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.1)

model_pipeline = Pipeline(
    steps=
    [
        ('col_formatter', cols_preprocessor)
    ]
)

x_train = x_train.fillna(method='ffill')
x_train_final = model_pipeline.fit_transform(x_train)


x_val_final = model_pipeline[0].transform(x_val)

y_val_final = (np.array(y_val)).reshape(-1, 1)
y_validation = num_imputer.fit_transform(y_val_final)

y_train_final = (np.array(y_train)).reshape(-1, 1)
y_training = num_imputer.transform(y_train_final)

my_model.fit(x_train_final, y_training, eval_set=[(x_val_final, y_validation)], verbose=0)

x_test_final = model_pipeline.fit_transform(x_test)

y_test = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\titanic\gender_submission.csv')
y_test_sel = y_test['Survived']
y_test_temp = (np.array(y_test_sel)).reshape(-1, 1)
y_test_final = num_imputer.fit_transform(y_test_temp)

y_test_predicted = my_model.predict(x_test_final)

scores = cross_val_score(my_model, x_test_final, y_test_predicted, cv=3)

final_test_set_x = pd.DataFrame(x_test_final[:,0],columns=['PassengerId'])
final_test_preds = pd.DataFrame(y_test_predicted,columns=['Survived'])


upload_preds = pd.concat([final_test_set_x, final_test_preds], axis=1)
upload_preds.to_csv(r'C:\Users\Mohit\Desktop\Titanic_Challenge_Submission.csv')