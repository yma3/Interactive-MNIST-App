import numpy as np
import tensorflow as tf
print(tf.__version__)

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255., X_test/255.

# Flatten Image

X_train = X_train.flatten().reshape(X_train.shape[0],-1)
print(X_train.shape)

model = TSNE(n_components=2, random_state=0, verbose=1)
tsne_data = model.fit_transform(X_train[:100,:])
tsne_data = np.vstack((tsne_data.T, y_train[:100])).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim 1", "Dim 2", "label"))

sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, "Dim 1", "Dim 2").add_legend()
plt.show()
