import numpy as np
import pandas as pd
#
# x=np.array([[1,2],[3,4]])
#
# y=np.expand_dims(x,axis=0)
#
# print(x.shape)
# print(y.shape)
#
# print(x.ndim,y.ndim)

array_test=pd.read_excel(r'C:\Users\Mohit\Desktop\test_array.xlsx')
arr_test=np.array(array_test)

print_arr=arr_test.reshape(3,1,10,1)

arra=np.expand_dims(arr_test,axis=0)
arraa=np.expand_dims(arr_test,axis=1)


# print(arr_test.shape)
# print(arra.shape)
# print(arra.shape)
print(arra[0].shape)
print(print_arr[0].shape)

# print(print_arr.shape)