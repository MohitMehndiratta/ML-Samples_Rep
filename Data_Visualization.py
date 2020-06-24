import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_iris=sn.load_dataset('iris')
plt.hist(df_iris['petal_width'])
plt.show()


mean = 50
sigma = 10

x_tst = range(1, 16)
x_test = np.random.normal(mean, sigma, 15)
y_test = np.random.normal(mean, sigma, 15).astype(int)

plt.plot(x_tst,y_test,color='green',ls='--',lw=4)
plt.plot(x_tst,y_test,color='green',marker=3,mew=10)

sales_in_delhi=[1000,2000,3000,4000,5000]
sales_in_mumbai=[1100,1500,2500,4800,2100]

colors=['red','green']
df_to_be_plotted=pd.DataFrame({"A":sales_in_delhi,"B":sales_in_mumbai})
df_to_be_plotted.plot(xticks=range(0,10),yticks=range(0,100,20),color=colors)

plt.bar(x_tst,y_test)

a=[10,12,20]
plt.pie(a,labels=["A","B","C"])

plt.show()


#------Plotting using object oriented way------------------
fig_object=plt.figure()
axes=fig_object.add_axes([.1,.1,1,1])

axes.grid()

axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_xticks(range(1,15))
axes.set_yticks(range(20,100,10))

axes.set_xlim([1,15])
axes.set_ylim([20,80])

axes.plot(x_tst,y_test)
plt.show()


#---- using subplots
fig_object_new=plt.figure()
fig_object_new,(axes1,axes2)=plt.subplots(1,2)
axes1.plot(x_tst)
axes2.plot(y_test)
plt.show()


#-------------------------------------------SEABORN PLOTS---------------------------------------------------

#--Numerical Plot ------
a = range(1, 16)
mean=50
sigma=100
b = np.random.normal(mean, sigma, 15)

df_tips=sn.load_dataset('tips')

sn.distplot(b)

sn.jointplot(a,b,kind="reg")
sn.jointplot(a,b,kind="hex")


sn.pairplot(df_iris,hue='species')


# ---Categorical Plots
sn.countplot('species',data=df_iris)

sn.boxplot('sepal_width','sepal_length',hue='species',data=df_iris)
print(df_iris.head())

sn.barplot('sepal_width','sepal_length',hue='species',data=df_iris)

sn.violinplot('sepal_width','sepal_length',hue='species',data=df_iris,palette='rainbow')

sn.violinplot(x='total_bill',y='day',data=df_tips,palette='')
plt.show()







