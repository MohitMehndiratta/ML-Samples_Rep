from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

diabetes_datset=datasets.load_diabetes()
print(diabetes_datset)

diabetes_x=diabetes_datset.data[:,np.newaxis,2]
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]

diabetes_y_train=diabetes_datset.target[:-30]
diabetes_y_test=diabetes_datset.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_predict=model.predict(diabetes_x_test)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predict)
plt.show()

# print(f"Mean Squared Error :  {mean_squared_error(diabetes_y_predict,diabetes_y_test)}")