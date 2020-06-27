from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

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
x_train_split, y_train_split, x_test_split, y_test_split = train_test_split(X_train, y_train, train_size=0.8,
                                                                            test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [col for col in x_train_split.columns if x_train_split[col].dtype=='object' and x_train_split[col].nunique()<10]
# print(categorical_cols)

numerical_cols = [col for col in x_train_split.columns if x_train_split[col].dtype=='int64']
# print(numerical_cols)

my_cols=categorical_cols+numerical_cols
# print(my_cols)

y=x_train_split[my_cols].copy()

print(y)