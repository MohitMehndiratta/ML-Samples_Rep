from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

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
x_train_split, x_test_split,y_train_split, y_test_split = train_test_split(X_train, y_train, train_size=0.8,
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
my_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('pre_processing', preprocessor),
])

# Preprocessing of training data, fit model
x_train_final=clf.fit_transform(x_train_n)
x_valid_n=clf.transform(x_valid_n)

my_model.fit(x_train_final,y_train_split)


# Preprocessing of validation data, get predictions
y_val_predicted=my_model.predict(x_valid_n)


#Evaluate the model
score = mean_absolute_error(y_val,y_val_predicted)
print('MAE:', score)
