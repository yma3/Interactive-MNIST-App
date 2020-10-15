import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255., X_test/255.
X_train = X_train.reshape((-1,28*28))
print(X_train.shape)
print(type(X_train))
pca = PCA(n_components=2)
# X_train_std = StandardScaler().fit_transform(X_train)

X_r = pca.fit(X_train).transform(X_train)
print(X_r.shape)
print(pca.components_[0])

fig, axes = plt.subplots(1,2, figsize=(9,18))

for i, ax in enumerate(axes.flat):
    ax.imshow((1-X_test[0]), alpha=1, cmap='gray')
    ax.imshow(pca.components_[i].reshape(28, 28), cmap='coolwarm', alpha=0.4)
    ax.grid(False)

plt.show()

SHOWN = 300
fig2 = plt.Figure()
plt.scatter(X_r[:SHOWN,0], X_r[:SHOWN,1], c=y_train[:SHOWN], cmap=plt.cm.get_cmap('tab10', 10))
plt.colorbar()
plt.show()
