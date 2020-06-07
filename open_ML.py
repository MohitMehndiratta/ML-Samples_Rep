from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt

pictorial_datset=fetch_openml('mnist_784')
test_data=pictorial_datset['data']
test_tbl=test_data[36000].reshape(28,28)

plt.imshow(test_tbl,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()

# test_data.reshape(28,28)
# plt.plot()
# plt.show()