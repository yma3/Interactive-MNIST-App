import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255., X_test/255.
X_train = X_train.flatten().reshape(-1, 28*28)
X_test = X_test.flatten().reshape(-1, 28*28)

model = KNeighborsClassifier(n_neighbors=5)
print("Fitting...")
model.fit(X_train, y_train)
print("Done Fitting")
print("Predicting...")
score = model.score(X_test, y_test)

print("accuracy=%2.f%%" % (score*100))

print("Saving...")
knnPickle = open('./models/mnist_knn.pkl', 'wb')
pickle.dump(model, knnPickle)
print("Saved.")
