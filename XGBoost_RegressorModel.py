from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# load the Data-------------------------------------------------------
X_full = pd.read_csv(r'C:\Users\Mohit\Desktop\Sample Files\melb_data.csv')

# Feature Engineering------------------------------------------------
X_full.dropna(axis=0, subset=['Price'], inplace=True)
y_full = X_full['Price']
X_full.drop(['Price'], axis=1, inplace=True)

# Break off validation set from training data -------------------------
X_train = X_full.iloc[:13480]
X_Val = X_full.iloc[13480:]
y_train = y_full.iloc[:13480]
y_val = y_full.iloc[13480:]

# Train Test Splitting -------------------------------------------------
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, train_size=0.8,
                                                                            test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [col for col in x_train_split.columns if
                    x_train_split[col].dtype == 'object' and x_train_split[col].nunique() < 10]

# Select numerical columns
numerical_cols = [col for col in x_train_split.columns if x_train_split[col].dtype == 'int64']

# Keep selected columns only
my_cols = categorical_cols + numerical_cols

x_train_n = x_train_split[my_cols].copy()
x_test_n = x_test_split[my_cols].copy()
x_valid_n = X_Val[my_cols].copy()
# x_test_new = x_test_split[my_cols].copy()

# Preprocessing for Numerical data
num_imputer = SimpleImputer(strategy='mean')

# Preprocessing for Categorical data
categorical_transformer = Pipeline([
    ('num', SimpleImputer(strategy='most_frequent')),
    ('cat', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical_processing', num_imputer, numerical_cols),
        ('categorical_processing', categorical_transformer, categorical_cols)
    ]
)

# Define model
# my_model = RandomForestRegressor(n_estimators=100, random_state=0)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.1)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('pre_processing', preprocessor),
])

clf2 = Pipeline(steps=[
    ('pre_processing', num_imputer),
])

# Preprocessing of training data, fit model
x_train_final = clf.fit_transform(x_train_n)
x_valid_n = clf.transform(x_valid_n)
x_test_n = clf.transform(x_test_split)

y_train_temp=np.array(y_train_split)
y_train_temp=y_train_temp.reshape(-1,1)
y_train_final = clf2.fit_transform(y_train_temp)


my_model.fit(x_train_final, y_train_split, early_stopping_rounds=5, eval_set=[(x_valid_n, y_val)])

params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
}

my_model_new = XGBRegressor()
random_search = RandomizedSearchCV(my_model_new, param_distributions=params, n_iter=5,  n_jobs=-1,
                                   cv=5, verbose=3)

random_search.fit(x_train_final, y_train_final)
# random_search.best_estimator_

my_model_advanced=XGBRegressor(n_estimators=1000, learning_rate=0.25)
my_model_advanced.fit(x_train_final, y_train_split, early_stopping_rounds=5, eval_set=[(x_valid_n, y_val)])


# Preprocessing of validation data, get predictions
y_test_predicted_model = my_model.predict(x_test_n)
y_test_predicted_model_advanced = my_model_advanced.predict(x_test_n)

# Evaluate the model
score_model1 = mean_absolute_error(y_test_split, y_test_predicted_model)
score_model2 = mean_absolute_error(y_test_split, y_test_predicted_model_advanced)

print('Model-1 MAE:', score_model1)
print('Model-2 MAE:', score_model2)