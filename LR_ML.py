from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


dataset_full=datasets.load_iris()
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

data_y_train=np.array(dataset_full['target']==2).astype(int)
data_x_train=dataset_full['data'][:,3:]

lgr_model=LogisticRegression()
lgr_model.fit(data_x_train,data_y_train)


data_x_test=np.linspace(0,3,1000).reshape(-1,1)
data_y_predicted=lgr_model.predict(data_x_test)
data_y_predicted_prob=lgr_model.predict_proba(data_x_test)[:,1]

# plt.plot(data_x_test,data_y_predicted_prob)
# plt.scatter(data_x_train,data_y_train)

plt.plot(data_x_test,data_y_predicted_prob)

plt.show()
