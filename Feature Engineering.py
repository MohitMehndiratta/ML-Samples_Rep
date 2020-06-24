import pandas as pd

comments = ["I am learning Python", "Python supports Machine Learning", "Python can handle BigData",
            "Python provides ML Algorithms"]

about_python = pd.DataFrame(comments)

# print(type(about_python),about_python)
# print(about_python.isnull())

python_df = pd.read_excel(r'C:\Users\Mohit\Desktop\About_Python.xlsx')

python_df_with_categories = pd.Categorical(python_df, categories=["Latest", "Newer", "Old", "Very Old", "Mostly used",
                                                                  "Well Supported"])
# print(python_df_with_categories)

python_versions = pd.DataFrame(python_df['Version'], dtype="category")
version_no = [0, 2, 3, 4]
version_comments = ["Very Old", "Old", "New"]

python_versions_desc = pd.cut(python_versions['Version'], version_no, labels=version_comments)
# print(python_versions_desc)

# Converting Categorical values to numerical
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
labels_encoded = encoder.fit_transform(comments)
# print(labels_encoded)

original_labels = encoder.inverse_transform(labels_encoded)
# print(original_labels)

# Getting Dummy Values for a datframe column
df_categories = pd.DataFrame({'data': [1, 2, 3, 4], 'Month': ['Jan', 'Feb', 'Mar', 'Apr']})
dummy_categories_dummy = pd.get_dummies(df_categories.data)

dummy_categories_dummy['Month'] = df_categories['Month']
dummy_categories_dummy['data'] = df_categories['data']
# print(dummy_categories_dummy)

df1 = pd.read_excel(r'C:\Users\Mohit\Desktop\About_Python.xlsx')
df2 = pd.read_excel(r'C:\Users\Mohit\Desktop\About_Py .xlsx')

df_concat = pd.concat([df1, df2])
# print(df_concat)
df_merge = pd.merge(df1, df2, on='Version')
# print(df_merge)


# using SK-Learn inbuilt methods--------------------------------------
from sklearn import preprocessing

data = {'price': [492, 286, 487, 519, 541, 429]}
price_frame = pd.DataFrame(data)

# Min-Max Normalization : Max value transforms to 1 and Min value transforms to 0
min_max_normalizer = preprocessing.MinMaxScaler()
normalized_data = min_max_normalizer.fit_transform(price_frame)
price_frame_normalized = pd.DataFrame(normalized_data)
# print(price_frame_normalized)

# Z-Score Normalization : Mean of the data is 0 and variance is 1
scaled_data = preprocessing.scale(price_frame)
# print(pd.DataFrame(scaled_data))

# Creating Word Counter
from sklearn.feature_extraction.text import CountVectorizer

comments = ['nice product', 'bad condition',
            'shipping was bad', 'great delivery'
            ]

norm_vector = CountVectorizer()
X_counter_train = norm_vector.fit_transform(comments)
test_df = pd.DataFrame(X_counter_train.toarray())

# Get all column name and integer index mapping
word_dict = dict((v, k) for k, v in norm_vector.vocabulary_.items())

# Now replace all integer columns with words
word_frame = test_df.rename(columns=word_dict)
# print(word_frame)

# Dimensionality Reduction
from sklearn.decomposition import PCA

pca = PCA()
word_frame_new = pd.DataFrame(pca.fit_transform(word_frame))
# print(word_frame_new)


import numpy as np

start_dt = '2020-06-01'
end_dt = '2020-06-30'

date_df = pd.DataFrame(pd.date_range(start_dt, end_dt), columns=['Dates'])
date_df['Day'] = date_df['Dates'].map(lambda x: x.day)
# date_df['WeekDay']=date_df['Dates'].map(lambda x : x.weekday_name)
# date_df['is_weekend'] = date_df['Dates'].map(lambda x:np.is_busday(x, weekmask="1111100"))
# print(date_df['is_weekend'])



a=pd.DataFrame(["2017-07-06","2017-07-07","2017-07-08"],columns=["Dates"])
a['is_weekend']=a['Dates'].map(lambda x:np.is_busday(x,weekmask="1111100"))
date_df['Dates'].astype('object')
# date_df['is_weekend']=date_df['Dates'].map(lambda x:np.is_busday(x,weekmask="1111100"))
# print(date_df)


print(date_df['Dates'].dtype)
print(a['Dates'].dtype)
