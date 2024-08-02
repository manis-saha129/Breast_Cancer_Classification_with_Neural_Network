import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
tf.random.set_seed(3)

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
# print the first 5 rows of the dataframe
print(data_frame.head())

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target
# data_frame.tail()
print(data_frame.tail())
# number of rows and columns in the dataset
print(data_frame.shape)
# getting some information about the data
print(data_frame.info())
# checking for missing values
print(data_frame.isnull().sum())
# statistical measure about the data
print(data_frame.describe())
# checking the distribution of Target Variable
print(data_frame['label'].value_counts())
# 0 ---> Malignant
# 1 ---> Benign
print(data_frame.groupby('label').mean())

# Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(X)
print(Y)

# Splitting the data into training data & testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Standardize the data
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)
print(X_train_std)

# Building the Neural Network
# setting up the layers of Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

# compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the Neural Network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Visualizing accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train data', 'validation data'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train data', 'validation data'], loc='upper right')
plt.show()

# Accuracy of the model on test data
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)

print(X_test_std.shape)
print(X_test_std[0])

Y_prediction = model.predict(X_test_std)
print(Y_prediction.shape)
print(Y_prediction[0])
print(X_test_std)
print(Y_prediction)

# model.predict() gives the prediction probability of each class for that data point
# argmax function
my_list = [0.25, 0.56]
index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)

# converting the prediction probability to class labels
Y_prediction_labels = [np.argmax(i) for i in Y_prediction]
print(Y_prediction_labels)

# Building the predictive system
input_data = (17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38,  17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189)
# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# standardizing the input data
input_data_std = scalar.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if prediction_label[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
