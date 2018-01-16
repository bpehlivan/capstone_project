#importing necessary libraries
import numpy as np
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, model_selection

# for mean squared error calculation
def mse( y_true, y_pred):
   y_err = y_true - y_pred
   y_sqr = y_err * y_err
   y_sum = np.sum(y_sqr)
   y_mse = y_sum / y_sqr.size
   return y_mse

#this is for showing real data
np.set_printoptions(suppress=True)

# this part is for readind .csv files and turning them into numpy arrays
data = genfromtxt('simple_pos.csv', delimiter=',', skip_header=1)

#this part slices arrays into two parts
#labels
#month(0),hour(1),maintenance(2),failure(3),weekday(4),temprature(5),position(6)
X = data[:, 0:6]

#feature
y = data[:, 6]

#scaling the label data for effective clustering
X = preprocessing.scale(X)
print(X.shape)
print(X)

#spliting data as training data and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#preparing classification object
mlp = MLPClassifier(random_state=0, hidden_layer_sizes=[100], max_iter=1)

# the part where training and observing error happens
# Note: to stop training at a certain point, it is required to create a keyboard
# interrupt to stop the whlie loop with CTRL-C
i = 1
cont = True
min_error = 1
while(i <= 2000):
    mlp.partial_fit(X_train, y_train, np.unique(y_train))
    y_pred = mlp.predict(X_train)
    mse_err = mse(y_train, y_pred)
    print("iteration: " + str(i) + " Mean Squared Error: " + str(mse_err))
    i = i + 1

#to see the accuracy of the prepared model
accuracy = mlp.score(X_test, y_test)
print(accuracy)
